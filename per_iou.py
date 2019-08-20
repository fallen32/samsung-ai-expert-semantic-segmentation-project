from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.distribute import distribution_strategy_context


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.

    If running in a `DistributionStrategy` context, the variable will be
    "sync on read". This means:

    *   The returned object will be a container with separate variables
      per replica of the model.

    *   When writing to the variable, e.g. using `assign_add` in a metric
      update, the update will be applied to the variable local to the
      replica.

    *   To get a metric's result value, we need to sum the variable values
      across the replicas before computing the final answer. Furthermore,
      the final answer should be computed once instead of in every
      replica. Both of these are accomplished by running the computation
      of the final result value inside
      `distribution_strategy_context.get_replica_context().merge_call(fn)`.
      Inside the `merge_call()`, ops are only added to the graph once
      and access to a sync on read variable in a computation returns
      the sum across all replicas.

    Args:
    shape: Shape of the created variable.
    dtype: Type of the created variable.
    validate_shape: (Optional) Whether shape validation is enabled for
      the created variable.
    name: (Optional) String name of the created variable.

    Returns:
    A (non-trainable) variable initialized to zero, or if inside a
    `DistributionStrategy` scope a sync on read variable container.
    """
    # Note that synchronization "ON_READ" implies trainable=False.
    return variable_scope.variable(
          lambda: array_ops.zeros(shape, dtype),
          trainable=False,
          collections=[
              ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
          ],
          validate_shape=validate_shape,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM,
          name=name)

def _aggregate_across_replicas(metrics_collections, metric_value_fn, *args):
  """Aggregate metric value across replicas."""
  def fn(distribution, *a):
    """Call `metric_value_fn` in the correct control flow context."""
    if hasattr(distribution.extended, '_outer_control_flow_context'):
      # If there was an outer context captured before this method was called,
      # then we enter that context to create the metric value op. If the
      # caputred context is `None`, ops.control_dependencies(None) gives the
      # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
      # captured context.
      # This special handling is needed because sometimes the metric is created
      # inside a while_loop (and perhaps a TPU rewrite context). But we don't
      # want the value op to be evaluated every step or on the TPU. So we
      # create it outside so that it can be evaluated at the end on the host,
      # once the update ops have been evaluted.

      # pylint: disable=protected-access
      if distribution.extended._outer_control_flow_context is None:
        with ops.control_dependencies(None):
          metric_value = metric_value_fn(distribution, *a)
      else:
        distribution.extended._outer_control_flow_context.Enter()
        metric_value = metric_value_fn(distribution, *a)
        distribution.extended._outer_control_flow_context.Exit()
        # pylint: enable=protected-access
    else:
      metric_value = metric_value_fn(distribution, *a)
    if metrics_collections:
      ops.add_to_collections(metrics_collections, metric_value)
    return metric_value

  return distribution_strategy_context.get_replica_context().merge_call(
      fn, args=args)

def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.

    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.

    Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).

    Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = metric_variable(
      [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')

    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = math_ops.cast(predictions, dtypes.int64)
    labels = math_ops.cast(labels, dtypes.int64)
    num_classes = math_ops.cast(num_classes, dtypes.int64)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = array_ops.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
      labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return total_cm, update_op


def per_class_iou(labels,
                  predictions,
                  num_classes,
                  weights=None,
                  metrics_collections=None,
                  updates_collections=None,
                  name=None):

  with variable_scope.variable_scope(name, 'mean_iou',
                                     (predictions, labels, weights)):
    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    total_cm, update_op = _streaming_confusion_matrix(labels, predictions,
                                                      num_classes, weights)

    def compute_per_class_iou(_, total_cm):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = math_ops.cast(
          math_ops.reduce_sum(total_cm, 0), dtypes.float32)
      sum_over_col = math_ops.cast(
          math_ops.reduce_sum(total_cm, 1), dtypes.float32)
      cm_diag = math_ops.cast(array_ops.diag_part(total_cm), dtypes.float32)
      denominator = sum_over_row + sum_over_col - cm_diag

      # The mean is only computed over classes that appear in the
      # label or prediction tensor. If the denominator is 0, we need to
      # ignore the class.
      num_valid_entries = math_ops.reduce_sum(
          math_ops.cast(
              math_ops.not_equal(denominator, 0), dtype=dtypes.float32))

      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = array_ops.where(
          math_ops.greater(denominator, 0), denominator,
          array_ops.ones_like(denominator))
      iou = math_ops.div(cm_diag, denominator)

      return iou

    # TODO(priyag): Use outside_compilation if in TPU context.
    iou_v = _aggregate_across_replicas(
        metrics_collections, compute_per_class_iou, total_cm)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return iou_v, update_op
