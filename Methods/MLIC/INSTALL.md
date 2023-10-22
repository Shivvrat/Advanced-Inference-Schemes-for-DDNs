# Installation

## Requirements
### Charades
There is a "joint\_ddn\_charades.yml" provided in a folder named requirements in the main directory. (We also provide these in the corresponding dataset folders)
Please use the following code snippet to create the environment. 

```
conda env create -f joint_ddn_charades.yml

```

You also need to download some packages required from Charades. These can be installation steps given in  [the PySlowFast Video Understanding repository](https://github.com/facebookresearch/slowfast). 
Add the slowfast directory to $PYTHONPATH.

```
export PYTHONPATH="/path/to/Charades/Joint Model/slowfast":$PYTHONPATH

``` 

### TaCOS and Wetlab
There is a "joint\_ddn\_tacos\_wetlab.yml" provided in a folder named requirements in the main directory. (We also provide these in the corresponding dataset folders)

Please use the following code snippet to create the environment. 

```
conda env create -f joint_ddn_tacos_wetlab.yml
```
