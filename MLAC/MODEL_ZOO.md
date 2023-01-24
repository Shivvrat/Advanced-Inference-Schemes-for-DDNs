# DDN Model Zoo
- For all the three Datasets we provide the jointly trained models (CNN and DN). These models are present in the directory - "models".
- We also used pre-trained models (both CNN and DN) to make the end to end learning faster. These models are located in folder named "pre-trained-models".
	1. **Charades**  
		- For Charades we used the SlowFast model which can be downloaded from the [PySlowFast repository](https://github.com/facebookresearch/SlowFast). 
		- We used the SlowFast R50 model trained on Charades using the standard training protocol. We updated the config file to accomodate DDNs. 
		- The config files are present in a folder called configs inside the Code directory. Please select the config which corresponds to the DN model you are using. 
	2. **TaCOS and Wetlab**	
		- The scripts to train the CNN model were adapted from this [repository](https://github.com/BartyzalRadek/Multi-label-Inception-net). We modified the code to accomodate the DDN and joint training. 
		- For TaCOS and Wetlab we used a pre-trained InceptionV3 model. 
		- Model and usage details are given in the repository we mentioned above. 
		- The config files are present in a folder called configs inside the corresponding code directory. Please select the config which corresponds to the DN model you are using. 
