# Getting Started with DDNs

This document provides a brief intro of running scripts to train and test the Joint DDN models for MLAC task. Updated Config files are provided in the directories.
Updated Config files (with values for joint learning) are provided in the respective directories. Note that all the details about the joint learning scripts are stored in the config files.

To show that DDNs generalize well we used three different datasets -

1. Charades
2. Wetlab
3. TACoS

## Charades

Please prepare the dataset following PySlowfast description for Charades.
You can look at the scripts for slowfast model from the PySlowFast repository.

### Train a Joint Model with pre-trained models and the loss from DDN to train the model jointly -

```
python tools_joint_learning/run_net.py \
--cfg configs/Charades/DN_NN_SLOWFAST_16x8_R50.yaml \
DATA.PATH_TO_DATA_DIR path_to_train_frame_list \
DATA.PATH_TO_DATA_DIR_TEST path_to_test_frame_list \
JOINT_LEARNING.MODEL_DIRECTORY path_to_trained_dn_model \
NUM_GPUS 1 \
TRAIN.BATCH_SIZE 16 \
JOINT_LEARNING.DN_TYPE "lr" \
JOINT_LEARNING.PRETRAINED True \
JOINT_LEARNING.TWO_LOSSES False \
```

You can give the type of DDN model you want to train using the JOINT_LEARNING.DN_TYPE.

### Gibbs Sampling for Jointly Learned Model -

```
python tools_joint_learning/run_net.py \
--cfg ./configs/Charades/DN_LR_SLOWFAST_16x8_R50.yaml \
DATA.PATH_TO_DATA_DIR_TEST path_to_test_frame_list \
NUM_GPUS 1 \
TRAIN.ENABLE False \
JOINT_LEARNING.DN_TYPE "lr" \
JOINT_LEARNING.PRETRAINED True \
TRAIN.CHECKPOINT_FILE_PATH path_to_your_slowfast_checkpoint \
JOINT_LEARNING.MODEL_DIRECTORY path_to_your_dn_checkpoint \
```

## TaCOS and Wetlab

### Train a joint Model with pre-trained models-

```
NAME="lr"
CURRENTDATE=$(date +"%Y-%m-%d_%T")

python src/model/retrain_without_test.py \
--bottleneck_dir= path_to_your_bottlenecks
--weights_and_biases_path=data/last_layer_model_parameters/ \
--model_dir=data/pre-trained_model/model_dir \
--date="${CURRENTDATE}" \
--output_graph="retrained_graph_${CURRENTDATE}.pb" \
--output_labels=retrained_labels.txt \
--model_name="${NAME}" \
--image_dir=path_to_your_images \
--image_labels_dir=path_to_your_image_labels
```

### Gibbs Sampling for Jointly Learned Model -

Please provide values to the \${NAME} used during training.

```
NAME="lr"

python ./src/model/evaluate_joint_model_test.py \
MODEL.SAVE_MODEL_PATH path_to_your_model \
MODEL.NAME "${NAME}"
```
