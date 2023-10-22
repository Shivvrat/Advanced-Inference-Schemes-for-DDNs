# Getting Started with DDNs

This document provides a brief intro of running scripts to train and test the Joint DDN models for MLIC task. 
Please prepare the dataset following Q2L and MSRN descriptions (which we used as feature extractors and baselines). 
Updated Config files (with values for joint learning) are provided in the respective directories. Note that all the details about the joint learning scripts are stored in the config files.

To show that DDNs generalize well we used three different datasets - 
1. COCO
2. PASCAL-VOC
3. NUS-WIDE

## COCO

### Train a Standard Model with pre-trained models and one loss - 

Here we will train a DDN-NN model jointly with the feature extractor. We have made a few changes to the orignal scripts provided by the authors of Q2L to take into account joint learning with DDN.
We need to get the outputs of the baseline model on the train set to train the pipeline models, thus we included this option (--val) to get the outputs on the train and test sets. 
The training code also does inference at the end to show how well the trained model has performed!

```
python main_mlc.py \
--backbone CvT_w24 \
--dataset_dir 'path/to/COCO14/' \
--dataname coco14 --batch-size 64 --val --workers 16 --print-freq 100 \
--config "path/to/config.json" \
--resume "path/to/checkpoint.pkl"  \
--output "path/to/output" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3717 \
--dtgfl \
--epochs num_epochs --lr 1e-6 --optim Adam --pretrained \
--cut_fact 0.5 \
--amp \
--weight-decay 1e-2 \
--dn_type nn
```

You can give the type of DN model you want to train using the JOINT_LEARNING.DN\_TYPE key.

### Test a Jointly Learned Model - 
We have made a few changes to the orignal scripts provided by the authors of Q2L to take into account joint learning with DDN

```
python q2l_infer.py -a "Q2L-CvT_w24-384" \
--dataset_dir 'path/to/COCO14/' --val \
--config "path/to/config.json" -b 48 \
--resume "path/to/jointly_trained/checkpoint.pkl" \
--output "path/to/output" \
--dist-url tcp://127.0.0.1:3451 \
--dn_type nn
```


## NUS

### Train a Standard Model with pre-trained models and two losses - 
The training code also does inference at the end to show how well the trained model has performed! Note that we are using a --val flag. 
We need to get the outputs of the baseline model on the train set to train the pipeline models, thus we included this option to get the outputs on the train and test sets. 
We need to get the outputs of the baseline model on the train set to get pre-trained DDN models before doing joint learning (we noted that if we take a pre-trained DDN, the convergence is a lot faster.). 
Thus to train the model on train set and test the model on the test set we use this flag (--val)

```
python3 nuswide_gcn.py data/nus \
--start-epoch 0 \
-b 16 --epochs 120 \
--lr 0.001 --val \
--image-size 448 \
--dn_type nn
```

### Test a Jointly Learned Model - 

```
python3 nuswide_gcn.py data/nus -e \
-b 16 \
--lr 0.0001 --val \
--image-size 448 \
--resume path/to/checkpoint/checkpoint.pth.tar \
--dn_type lr
```


## VOC
The training code also does inference at the end to show how well the trained model has performed!
We need to get the outputs of the baseline model on the train set to train the pipeline models, thus we included this option to get the outputs on the train and test sets. 
Similar to the scripts provided for NUS, note that we are using a --val flag. The reason for using the --val flag is same here as well.  

### Train a Standard Model with pre-trained models and two losses - 

```
python3 joint_voc.py data/voc \
--start-epoch 0 \
-b 16 --epochs 160 \
--lr 0.001 --val \
--image-size 448 \
--dn_type nn
```

### Test a Jointly Learned Model - 

```
python3 joint_voc.py data/voc -e \
-b 16 \
--lr 0.0001 --val \
--image-size 448 \
--resume path/to/checkpoint/checkpoint.pth.tar \
--dn_type lr
```