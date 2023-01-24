from utils import create_directory
from dn.inference_on_dn import gibbs_sampling, get_model
from loguru import logger
import wandb
from tools.parser import load_cfgs, parse_args, convert_to_dict
import re
import tensorflow as tf
import os
import pickle
import pprint
import time

import numpy
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.metrics import hamming_loss, accuracy_score, coverage_error, label_ranking_average_precision_score, \
    label_ranking_loss, jaccard_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_labels(label_file):
    labels = np.genfromtxt(label_file, dtype=float, delimiter=",").astype(int)
    return labels


def get_frame_name(x):
    # return math.sqrt(x)
    if type(x) is not str:
        raise TypeError("Input give to get_frame_name is not string")
    out = str(x[:-4])
    return out


def predict_labels(image_paths: str, model_path: str):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    predicted_labels = []
    sorted_image_paths = natsorted(image_paths)
    images_names = []
    for image_path in sorted_image_paths:
        image_name = image_path.split("/")[-1]
        images_names.append(image_name)
        print("Running for file - %s" % image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        # Loads label file, strips off carriage return
        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            if len(predicted_labels) == 0:
                predicted_labels = numpy.append(predicted_labels, predictions)
            else:
                predicted_labels = numpy.vstack(
                    [predicted_labels, predictions])

    return numpy.array(predicted_labels), images_names


@logger.catch()
def main():
    wandb.init(project="Joint Learning - Inference - Wetlab")
    args = parse_args()
    cfg = load_cfgs(args)
    wandb.config.update(cfg)
    dn_models = get_model(cfg)
    logger.info("Inference with config:")
    logger.info(pprint.pformat(cfg))
    import sys
    # if cfg.DEBUG:
    # Provide the image paths here
    image_path_given = ""
    # image_path_given = "./test/test_data_for_pgm_tranX/"
    image_paths = [os.path.join(image_path_given, img)
                   for img in os.listdir(image_path_given) if '.jpg' in img]
    if cfg.DEBUG:
        image_paths = image_paths[:5]
    logger.info(
        f"Using the trained CNN located at {cfg.MODEL.SAVE_MODEL_PATH}cnn.pb")
    y_predicted, images_names = predict_labels(
        image_paths, cfg.MODEL.SAVE_MODEL_PATH + "cnn.pb")
    true_label_path = "./data/external/true_labels/wetlab.test.ground"
    true_label = get_labels(true_label_path)
    true_label_frames = np.genfromtxt("./data/external/true_labels/test_frames.txt", dtype=str).reshape(
            (true_label.shape[0], 1))
    num_examples_in_test_set, num_labels = y_predicted.shape
    #     true_label_frames = np.genfromtxt("./data/external/true_labels/test_labels/test_frames.txt", dtype=str)[:5].reshape(
    #             (num_examples_in_test_set, 1))
    #     true_label = true_label[:5]
    output_CNN_frames = images_names
    output_CNN_frames = np.array([get_frame_name(
        xi) for xi in output_CNN_frames]).reshape((num_examples_in_test_set, 1))
    true_label_with_frames = pd.DataFrame(
        np.hstack((true_label_frames, true_label)))
    output_CNN_with_frames = pd.DataFrame(
        np.hstack((output_CNN_frames, y_predicted)))
    res = true_label_with_frames.merge(output_CNN_with_frames, on=0).values
    true_label = res[:, 1:num_labels + 1].astype(int)
    output_of_CNN = res[:, num_labels + 1:].astype(float)
    frame_names = res[:, 0]
    total_outputs_test = {"ground_truth": true_label,
                          "output of cnn": output_of_CNN, "frame_names": frame_names}
    from utils import save_data_to_pickle
    from datetime import datetime
    date_time = str(datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    save_path = os.path.join(cfg.OUTPUT_PATH, date_time, cfg.MODEL.NAME)
    create_directory(save_path)

    if not cfg.DEBUG:
        save_data_to_pickle(total_outputs_test, f'{save_path}/cnn_outputs.pkl')
    if true_label.shape[1] == 0 or cfg.DEBUG:
        logger.info(frame_names)
        logger.info(images_names)
        logger.info(true_label)
        logger.info(f"The final dataframe - {res}")
        if not cfg.DEBUG:
            # Only using the first few examples for debug mode
            y_true = true_label[:y_predicted.shape[0]]
            raise ValueError("The array is empty")

    logger.info(f"Successfully saved prediction results to {save_path}")
    logger.info(
        f"Importing Dependency Network Models from {cfg.MODEL.SAVE_MODEL_PATH}")
    logger.info(
        "Doing Inference on DN networks using the outputs created by CNN")
    wandb.alert(title="Testing CNN Complete",
                text="Completed training the CNN model")
    gibbs_sampling(true_label, output_of_CNN, dn_models, cfg.DN_INFERENCE.NUM_SAMPLES, save_path,
                   cfg.DN_INFERENCE.BATCH_SIZE, logger, cfg)


if __name__ == "__main__":
    # 2nd argument is the model path
    main()
