import tensorflow as tf


class IOU:
    def __init__(self, num_classes):
        """
        :param num_classes: number of classes
        """
        self.num_classes = num_classes

    def calc_batch_cm(self, labels, predictions):
        """
        :param labels: true labels of each pixels. argmax should be performed before use this function.
                       dims = ( batch, height, width )
        :param predictions: predicted labels of each pixels. argmax should be performed before use this function.
                       dims = ( batch, height, width )
        :return: confusion_matrix: calculated temporal confusion_matrix
        """
        labels_flatten = tf.reshape(labels, [-1])
        pred_flatten = tf.reshape(predictions, [-1])
        confusion_matrix = tf.math.confusion_matrix(labels_flatten, pred_flatten, self.num_classes)

        return confusion_matrix

    def iou_class(self, confusion_matrix):
        """
        :param confusion_matrix: running confusion matrix
        :return: iou: calculated IOU per class.
        """
        pred_areas = tf.cast(tf.reduce_sum(confusion_matrix, axis=0), tf.float32)

        labels_areas = tf.cast(tf.reduce_sum(confusion_matrix, axis=1), tf.float32)

        intersection = tf.cast(tf.linalg.diag_part(confusion_matrix), tf.float32)

        union = pred_areas + labels_areas - intersection

        union = tf.where_v2(tf.greater(union, 0), union, tf.ones_like(union))

        iou = tf.div(intersection, union)

        return iou

    def iou_label_list(self):
        labels = ['Unknown', 'Road', 'Sidewalk',
                  'Building', 'Wall', 'Fence',
                  'Pole', 'Traffic light', 'Traffic Sign',
                  'Vegetation', 'Terrain', 'Sky',
                  'Person', 'Rider', 'Car',
                  'Truck', 'Bus', 'Train',
                  'Motorcycle', 'Bicycle']

        return labels