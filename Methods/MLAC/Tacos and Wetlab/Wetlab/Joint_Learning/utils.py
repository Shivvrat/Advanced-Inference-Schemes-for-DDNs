import os
import pickle

import numpy


def delete_given_file(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)


def get_date_as_string():
    from datetime import datetime
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    return date


# pip install utilsd

def import_result_from_MAR_file(mar_file_location):
    with open(mar_file_location) as file:
        lines = file.readlines()


def get_data_from_markov_network_train_data(location, logger):
    """
    Return the actual labels and joint_learning_cnn_with_mn predictions in sequence
    Also takes in account missing data (imported as NaN values)
    :param location:
    :param logger:
    :return:
    """
    import numpy as np
    import pandas as pd
    data = pd.read_csv(location, header=None, dtype=float).to_numpy()
    logger.info(f"The shape of the data is - {data.shape}")
    # test_CNN_predictions = pd.read_csv(test_CNN_predictions_location, sep=",", header=None)
    # logger.info(args.test_CNN_predictions_and_true_labels_name )
    # logger.info(test_CNN_predictions_and_true_labels)
    actual_labels, CNN_predictions = data[:, :data.shape[1] // 2], data[:, data.shape[1] // 2:]
    return actual_labels, CNN_predictions


def get_data_from_slowfast_output_pkl(train_CNN_predictions_and_true_labels_location):
    """
    Get the predictions and actual_output from slowfast output (pkl file)
    @param train_CNN_predictions_and_true_labels_location: Location of pkl file
    @return: predictions of CNN and true label
    """
    f = open(train_CNN_predictions_and_true_labels_location, 'rb')
    import pickle
    train_predictions_and_actual_output = pickle.load(f)
    f.close()
    predictions, actual_output = train_predictions_and_actual_output[0].numpy(), train_predictions_and_actual_output[
        1].numpy()
    return predictions, actual_output


def get_data_from_acar_output_csv(train_CNN_predictions_and_true_labels_location):
    """
    Get the predictions and actual_output from slowfast output (pkl file)
    @param train_CNN_predictions_and_true_labels_location: Location of pkl file
    @return: predictions of CNN and true label
    """
    import numpy as np
    test_predictions_and_actual_output = np.genfromtxt(train_CNN_predictions_and_true_labels_location, delimiter=',')
    num_true_labels = test_predictions_and_actual_output.shape[1] // 2
    test_actual_labels = test_predictions_and_actual_output[:, :num_true_labels]
    test_CNN_predictions = test_predictions_and_actual_output[:, num_true_labels:]
    return test_CNN_predictions, test_actual_labels


def get_data_from_custom_inception(path):
    with open(path, 'rb') as f:
        train_predictions_and_actual_output = pickle.load(f)
    ground_truth = train_predictions_and_actual_output['ground']
    cnn_predictions = train_predictions_and_actual_output['joint_learning_cnn_with_mn']
    return ground_truth, cnn_predictions


def create_directory(directory_path):
    """
    Create a directory given the directory path
    @param directory_path: The path of directory to be created
    """
    # Make output dir and its parents if they do not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_index_given_truth_value(truth_value):
    """
    Get the index in a probability vector given the truth values
    @param truth_value: the truth values from which index is to be infered
    @type truth_value: ndarray or list
    @return: Return the index in a probability vector
    @rtype:
    """
    index = 0
    count = 0
    for each_val in reversed(truth_value):
        index += 2 ** count * int(each_val)
        count += 1
    return index


def get_truth_values_given_index(variables, index_value, cardinalities):
    """
    Gives the truth value (values in tuple) for given index of the array
    @param variables: list containing all the variables in the graph which are in the factor
    @param index_value: Index value for which the tuple is to be founded
    @param cardinalities: Cardinalities of the given variables
    @return: the tuple for corresponding index value
    """
    number = 0
    truth_table_value = []
    while number < len(variables):
        truth_table_value.append(int(index_value // numpy.prod(cardinalities[number + 1:])))
        index_value = index_value - (index_value // numpy.prod(cardinalities[number + 1:])) * numpy.prod(
            cardinalities[number + 1:])
        number += 1
    return truth_table_value


def convert_to_log_space(distribution_array):
    from numpy import log10
    new_distribution_array = []
    for each in distribution_array:
        if each is not list:
            each_new = log10(each)
        else:
            each_new = [(log10(each_value)) for each_value in each]
        new_distribution_array.append(each_new)
    return new_distribution_array


def import_from_pickle_file(file_location):
    with open(file_location, 'rb') as file:
        object_file = pickle.load(file)
    return object_file


def save_data_to_pickle(data, location):
    with open(location, "wb") as filehandler:
        pickle.dump(data, filehandler)


def convert_to_exponent_space(distribution_array):
    new_distribution_array = []
    for each in distribution_array:
        if each is not list:
            each_new = 10 ** each
        else:
            each_new = [10 ** each_value for each_value in each]
        new_distribution_array.append(each_new)
    return new_distribution_array


def calculate_evaluation_metrics(y_true, y_pred_probability, threshold):
    """
    Calculate the Evaluation Metrics given the true label and sigmoid output for a multi label classification task
    @param y_true: Vector of true values
    @type y_true: array
    @param y_pred_probability: Vector of output predictions
    @type y_pred_probability: array
    @return: Multiple metrics
    @rtype:
    """
    from sklearn.metrics import hamming_loss
    from sklearn.metrics import label_ranking_loss, coverage_error, label_ranking_average_precision_score, \
        accuracy_score, average_precision_score, \
        jaccard_score
    y_pred = numpy.copy(y_pred_probability)
    y_pred_probability = numpy.array(y_pred_probability)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred_probability = y_pred_probability.reshape(y_pred_probability.shape[0], -1)
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = 0
    y_true = numpy.array(y_true)
    # result = (y_pred < 0.5) * y_pred
    hammingLoss = hamming_loss(y_true, y_pred)
    coverage = coverage_error(y_true, y_pred_probability)
    ranking_loss = label_ranking_loss(y_true, y_pred_probability)
    label_ranking_average_precision_score = label_ranking_average_precision_score(y_true, y_pred_probability)
    average_precision = average_precision_score(y_true, y_pred_probability)
    subset_accuracy = accuracy_score(y_true, y_pred)
    jaccard_score_val = jaccard_score(y_true, y_pred, average='samples')
    return hammingLoss, coverage, ranking_loss, average_precision, subset_accuracy, jaccard_score_val, label_ranking_average_precision_score


from loguru import logger


def print_and_save_eval_metrics(metrics, output_filename, logger=logger):
    """
    Save the multi label classification metrics

    """
    hammingLoss, coverage, ranking_loss, average_precision, subset_accuracy, jaccard_score_val, label_ranking_average_precision_score = metrics
    logger.info(f"Hamming Loss : {str(hammingLoss)}")
    logger.info("Coverage : " + str(coverage))
    logger.info("Ranking Loss : " + str(ranking_loss))
    logger.info("Average Precision : " + str(average_precision))
    logger.info("Subset Accuracy : " + str(subset_accuracy))
    logger.info("Jaccard Score : " + str(jaccard_score_val))
    logger.info("Label Ranking Average Precision Score : " + str(label_ranking_average_precision_score))
    eval_dict = {"Hamming Loss": str(hammingLoss), "Coverage": str(coverage), "Ranking Loss": str(ranking_loss),
                 "Average Precision": str(average_precision), "Subset Accuracy": str(subset_accuracy),
                 "Jaccard Score": str(jaccard_score_val),
                 "Label Ranking Precision Score": str(label_ranking_average_precision_score)}

    import pandas as pd
    (pd.DataFrame.from_dict(data=eval_dict, orient='index').to_csv(output_filename, header=False))


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    import GPUtil
    GPUtil.showUtilization()
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    import psutil
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def load_image_given_path(path, shape=(299, 299), interpolation='nearest'):
    """
    Load an image and resize it using a specific interpolation method
    @param path:
    @type path:
    @param shape:
    @type shape:
    @param interpolation:
    @type interpolation:
    @return:
    @rtype:
    """
    import PIL
    import numpy as np
    img = PIL.Image.open(path)
    if interpolation == 'nearest':
        img = img.resize(shape)
    elif interpolation == 'bilinear':
        img = img.resize(shape, PIL.Image.BILINEAR)
    img = np.asarray(img) / 255
    img = img.reshape((1, *shape, 3))
    return img


def import_edge_list(edge_list_file_path):
    from numpy import genfromtxt
    edge_list = genfromtxt(edge_list_file_path, delimiter=" ", skip_header=True, dtype=str)
    return edge_list


def generate_data_array_given_image_path(path, onehot=False, image_shape=(299, 299)):
    """
    Generate two vectors one containing the images as a vector and one containing image paths
    @param path: Path of the image directory
    @type path:
    @param onehot: If output is required to be one hot vector
    @type onehot:
    @param image_shape: Shape of the image
    @type image_shape:
    @return:two vectors one containing the images as a vector and one containing image paths
    @rtype:
    """
    import numpy as np
    x = []
    y = []
    for i, category in enumerate(os.listdir(path)):
        for image in os.listdir(os.path.join(path, category)):
            image_path = os.path.join(path, category, image)
            try:
                x.append(load_image_given_path(image_path, image_shape))
                y.append(i)
            except Exception as e:
                print(e)
    x = np.concatenate(x)
    y = np.array(y)
    if onehot:
        from tensorflow.python.keras.utils.np_utils import to_categorical
        y = to_categorical(y)
    return x, y


def import_all_mn(potentials_location):
    potential_locations_dict = {}
    struct_locations_dict = {}
    potentials_dict = {}
    pll_dict = {}
    struct_dict = {}
    num_markov_networks = 0
    for (dirpath, dirnames, filenames) in os.walk(potentials_location):
        for each_file_name in filenames:
            path_to_current_file = os.path.join(dirpath, each_file_name)
            if "struct" in each_file_name:
                struct_dict[dirpath[-1]] = utils.import_edge_list(path_to_current_file)
                if dirpath not in struct_locations_dict:
                    struct_locations_dict[dirpath[-1]] = [path_to_current_file]
                else:
                    struct_locations_dict[dirpath[-1]].append(path_to_current_file)
            elif "potential" in each_file_name:
                num_markov_networks += 1
                potentials_dict[dirpath[-1]] = import_potentials(path_to_current_file)
                if dirpath not in potential_locations_dict:
                    potential_locations_dict[dirpath[-1]] = [path_to_current_file]
                else:
                    potential_locations_dict[dirpath[-1]].append(path_to_current_file)
            elif "pll_values" in each_file_name:
                pll_dict[dirpath[-1]] = utils.import_from_pickle_file(path_to_current_file)
    return potential_locations_dict, struct_locations_dict, potentials_dict, pll_dict, struct_dict, num_markov_networks
