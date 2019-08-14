import tensorflow as tf


class IOU:
    def __init__(self, num_classes):
        """
        :param num_classes: number of classes
        """
        self.num_classes = num_classes

    def iou_class(self, labels, predictions):
        """
        :param labels: true labels of each pixels. argmax should be performed before use this function.
                       dims = ( batch, height, width )
        :param predictions: predicted labels of each pixels. argmax should be performed before use this function.
                       dims = ( batch, height, width )
        :param num_batches: number of batch images
        :return: iou: calculated IOU per class.
        """
        labels_flatten = tf.reshape(labels, [-1])
        pred_flatten = tf.reshape(predictions, [-1])
        confusion_matrices = tf.math.confusion_matrix(labels_flatten, pred_flatten, self.num_classes)

        pred_areas = tf.cast(tf.reduce_sum(confusion_matrices, axis=0), tf.float32)

        labels_areas = tf.cast(tf.reduce_sum(confusion_matrices, axis=1), tf.float32)

        intersection = tf.cast(tf.linalg.diag_part(confusion_matrices), tf.float32)

        union = pred_areas + labels_areas - intersection

        union = tf.where_v2(tf.greater(union, 0), union, tf.ones_like(union))

        iou = tf.div(intersection, union)

        return iou
