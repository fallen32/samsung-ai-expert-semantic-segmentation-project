from deeplabV2_sep_mixup_tensorflow import FCN8s
from data_generator.batch_generator import BatchGenerator
from helpers.visualization_utils import print_segmentation_onto_image, create_video_from_images
from cityscapesscripts.helpers.labels import TRAINIDS_TO_COLORS_DICT, TRAINIDS_TO_RGBA_DICT

from math import ceil
import os
import time
import matplotlib.pyplot as plt

# TODO: Set the paths to the images.
# dataset_root_dir = '../data/Cityscapes_small_128_256'
dataset_root_dir = 'data/Cityscapes_small'

train_images = os.path.join(dataset_root_dir, 'leftImg8bit/train/')
val_images = os.path.join(dataset_root_dir, 'leftImg8bit/val/')
test_images = os.path.join(dataset_root_dir, 'leftImg8bit/test/')

# TODO: Set the paths to the ground truth images.
train_gt = os.path.join(dataset_root_dir, 'gtFine/train/')
val_gt = os.path.join(dataset_root_dir, 'gtFine/val/')

# Put the paths to the datasets in lists, because that's what `BatchGenerator` requires as input.
train_image_dirs = [train_images]
train_ground_truth_dirs = [train_gt]
val_image_dirs = [val_images]
val_ground_truth_dirs = [val_gt]

num_classes = 20 # TODO: Set the number of segmentation classes.

train_dataset = BatchGenerator(image_dirs=train_image_dirs,
                               image_file_extension='png',
                               ground_truth_dirs=train_ground_truth_dirs,
                               image_name_split_separator='leftImg8bit',
                               ground_truth_suffix='gtFine_labelIds',
                               check_existence=True,
                               num_classes=num_classes)
num_train_images = train_dataset.get_num_files()
print("Size of training dataset: ", num_train_images, " images")

val_dataset = BatchGenerator(image_dirs=val_image_dirs,
                             image_file_extension='png',
                             ground_truth_dirs=val_ground_truth_dirs,
                             image_name_split_separator='leftImg8bit',
                             ground_truth_suffix='gtFine_labelIds',
                             check_existence=True,
                             num_classes=num_classes)


num_val_images = val_dataset.get_num_files()
print("Size of validation dataset: ", num_val_images, " images")

# TODO: Set the batch size. I'll use the same batch size for both generators here.
batch_size = 4

train_generator1 = train_dataset.generate(batch_size=batch_size,
                                          convert_colors_to_ids=False,
                                          convert_ids_to_ids=False,
                                          convert_to_one_hot=True,
                                          void_class_id=None,
                                          random_crop=False,
                                          crop=False,
                                          resize=False,
                                          brightness=False,
                                          flip=0.5,
                                          translate=False,
                                          scale=False,
                                          gray=False,
                                          to_disk=False,
                                          shuffle=True)

train_generator2 = train_dataset.generate(batch_size=batch_size,
                                          convert_colors_to_ids=False,
                                          convert_ids_to_ids=False,
                                          convert_to_one_hot=True,
                                          void_class_id=None,
                                          random_crop=False,
                                          crop=False,
                                          resize=False,
                                          brightness=False,
                                          flip=0.5,
                                          translate=False,
                                          scale=False,
                                          gray=False,
                                          to_disk=False,
                                          shuffle=True)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     convert_colors_to_ids=False,
                                     convert_ids_to_ids=False,
                                     convert_to_one_hot=True,
                                     void_class_id=None,
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     gray=False,
                                     to_disk=False,
                                     shuffle=True)

model = FCN8s(model_load_dir=None,
              tags=None,
              vgg16_dir='data/VGG-16_mod2FCN_ImageNet-Classification',
              num_classes=num_classes,
              variables_load_dir=None)

epochs = 20  # TODO: Set the number of epochs to train for.


# TODO: Define a learning rate schedule function to be passed to the `train()` method.
def learning_rate_schedule(step):
    if step <= 12000:
        return 0.0001
    elif 12000 < step <= 24000:
        return 0.00001
    elif 24000 < step <= 48000:
        return 0.000003
    else:
        return 0.000001


model.train(train_generator1=train_generator1,
            epochs=epochs,
            steps_per_epoch=ceil(num_train_images / batch_size),
            learning_rate_schedule=learning_rate_schedule,
            keep_prob=0.5,
            l2_regularization=0.0,
            eval_dataset='val',
            eval_frequency=1,
            val_generator=val_generator,
            val_steps=ceil(num_val_images / batch_size),
            metrics={'loss', 'mean_iou', 'accuracy'},
            save_during_training=False,
            save_dir='cityscapes_model',
            save_best_only=True,
            save_tags=['default'],
            save_name='(batch-size-4)',
            save_frequency=2,
            saver='saved_model',
            monitor='loss',
            record_summaries=False,
            summaries_frequency=10,
            summaries_dir='tensorboard_log/cityscapes',
            summaries_name='deeplab_01',
            training_loss_display_averaging=3)

"""
labels_argmax = tf.argmax(self.labels, axis=-1, name='labels_argmax', output_type=tf.int64)
labels = self.sess.run(labels_argmax, feed_dict={self.image_input: batch_images,
                                                      self.labels: batch_labels,
                                                      self.learning_rate: learning_rate,
                                                      self.keep_prob: keep_prob,
                                                      self.l2_regularization_rate: l2_regularization})
predictions = self.sess.run(self.predictions_argmax, feed_dict={self.image_input: batch_images,
                                                      self.labels: batch_labels,
                                                      self.learning_rate: learning_rate,
                                                      self.keep_prob: keep_prob,
                                                      self.l2_regularization_rate: l2_regularization})
from metric_iou import IOU
test = IOU(self.num_classes)
per_class = test.iou_class(labels, predictions, 4)
result = self.sess.run(per_class, feed_dict={self.image_input: batch_images,
                                                      self.labels: batch_labels,
                                                      self.learning_rate: learning_rate,
                                                      self.keep_prob: keep_prob,
                                                      self.l2_regularization_rate: l2_regularization})
                                                      
a, b, c, d = self.sess.run([cur_iou_class, union, num_valid, intersection], feed_dict={self.image_input: batch_images,
                                                      self.labels: batch_labels,
                                                      self.keep_prob: 1.0})  
"""