# DDN Model Zoo for MLIC
- For all the three Datasets we will provide the jointly trained models (CNN and DN). These models are present in the directory - "models". 
- We also used pre-trained models (both CNN and DN) to make the end to end learning faster. These models are located in folder named "pre-trained-models".
	1. **MS-COCO**  
		- For COCO we used the Queries2Label model as baseline which can be downloaded from the [Q2L repository](https://github.com/SlongLiu/query2labels). 
		- We used the Q2L-CvT_w24-384 trained on MS-COCO. 
		- The config files are present in a folder called configs inside the Code directory. Please select the config which corresponds to the DN model you are using.
		- We also need to download the data folder in the repository for some files to make Q2L work. 
	2. **PASCAL-VOC and NUS-WIDE**	
. 		- For both these datsets we used the MSRN model as baseline which can be downloaded from the [MSRN.pytorch repository](https://github.com/chehao2628/MSRN). 
		- The baseline models can be downloaded by following the instructions provided in the MSRN.pytorch repository.
		- The config files are present in a folder called configs inside the corresponding code directory. Please select the config which corresponds to the DN model you are using. 
