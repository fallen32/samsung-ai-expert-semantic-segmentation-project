import tensorflow as tf

vgg16_tag = 'vgg16'
vgg16_dir = 'data/VGG-16_mod2FCN_ImageNet-Classification'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

tf.saved_model.loader.load(sess=sess, tags=[vgg16_tag], export_dir=vgg16_dir)
graph = tf.get_default_graph()

tensor_names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
