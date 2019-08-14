import tensorflow as tf


class IOU:
    def __init__(self, num_classes):
        """
        :param num_classes: number of classes
        """
        self.num_classes = num_classes

    def iou_class(self, labels, predictions, num_batches):
        """
        :param labels: true labels of each pixels. argmax should be performed before use this function.
                       dims = ( batch, height, width )
        :param predictions: predicted labels of each pixels. argmax should be performed before use this function.
                       dims = ( batch, height, width )
        :param num_batches: number of batch images
        :return: iou_class: calculated IOU per class.
        """
        labels_flatten = tf.layers.flatten(inputs=labels)
        pred_flatten = tf.layers.flatten(inputs=predictions)
        confusion_matrices = [tf.math.confusion_matrix(labels_flatten[i], pred_flatten[i], self.num_classes)
                              for i in range(num_batches)]

        pred_areas = tf.convert_to_tensor([tf.reduce_sum(confusion_matrices[i], axis=0) for i in range(num_batches)],
                                          dtype=tf.float32)

        labels_areas = tf.convert_to_tensor([tf.reduce_sum(confusion_matrices[i], axis=1) for i in range(num_batches)],
                                            dtype=tf.float32) + 1e-6    # for stability

        intersection = tf.convert_to_tensor([tf.linalg.diag_part(confusion_matrices[i]) for i in range(num_batches)],
                                            dtype=tf.float32)

        union = pred_areas + labels_areas - intersection

        iou_class = intersection / union

        non_zero = tf.convert_to_tensor([tf.count_nonzero(iou_class[:, i]) for i in range(self.num_classes)],
                                        dtype=tf.float32) + 1e-6

        iou_class = tf.reduce_sum(iou_class, axis=0) / non_zero

        return iou_class
