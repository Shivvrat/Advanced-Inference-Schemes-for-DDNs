# DDN for COCO

## Introduction

The goal of DDN is to provide a faster training, light-weight codebase that improve on the baseline deep neural networks based video backbones for multi-label activity classification.

## Model Zoo and Baselines

Please download the trained models available in the Query2Labels [query2labels repository](https://github.com/SlongLiu/query2labels). We used the Q2L-CvT_w24-384 availble in the [Pretrained Models section](https://github.com/SlongLiu/query2labels#pretrianed-models). 

## Installation

- Please use the corresponding yml file to create the environment required to run the code. 
- To download the datasets for COCO dataset please follow the instructions given on query2label repository. Note that we use the MS-COCO 2014 datset.
Please follow all the details on query2labels repository to setup the feature extractor correctly.
- We also need to download some files that are required by Q2L. Please download them from the [Q2L repository](https://github.com/SlongLiu/query2labels). Place the folder named data inside the ddn-joint folder. 
