import os

import tensorflow as tf
from tensorflow.python.platform import gfile


def get_values_of_weights_ands_biases(location=os.path.join('./data/pre-trained_model/', 'retrained_graph_250000.pb')):
    with tf.Session() as sess:
        # model_filename = os.path.join(
        #     './data/pre-trained_model/', 'retrained_graph_250000.pb')
        model_filename = location
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            sess.graph.as_default()
            constant_values = {}
            with tf.Session() as sess:
                constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
                for constant_op in constant_ops:
                    constant_values[constant_op.name] = sess.run(constant_op.outputs[0])
            # print(constant_values['final_training_ops/biases/final_biases'])
            # print(constant_values['final_training_ops/weights/final_weights'])
            FINAL_BIASES = constant_values['final_training_ops/biases/final_biases']
            FINAL_WEIGHTS = constant_values['final_training_ops/weights/final_weights']
    return FINAL_BIASES, FINAL_WEIGHTS


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


from numpy import save
import sys
FINAL_BIASES, FINAL_WEIGHTS = get_values_of_weights_ands_biases(sys.argv[1])
path = f'./data/last_layer_model_parameters/{sys.argv[2]}/'
ensure_dir_exists(path)
save(f'{path}final_biases', FINAL_BIASES)
save(f'{path}final_weights', FINAL_WEIGHTS)
