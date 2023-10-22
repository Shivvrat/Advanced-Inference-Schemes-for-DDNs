# Advanced *MPE* Inference Schemes for Dependency Networks

In this directory, we provide two distinct inference schemes that are essential to our research:

### 1. Local Search-Based Methods

These methods leverage local search techniques to optimize the inference process within our Deep Dependency Networks. Local search algorithms focus on finding the best solution in the vicinity of an initial solution.

#### Usage

To apply Local Search-Based Methods to your multi-label classification task, follow these steps:

1. **Navigate to the Directory** : Locate the `run_local_search.py` script within this directory.
2. **Configuration** : Modify the script's configuration file to suit your specific task and dataset requirements.
   1. `--dataset`: Name of the dataset you want to use.
   2. `--dn_type`: Name of the Dependency Network type (e.g., "lr").
   3. `--search_method`: Choose the search method from the following options:
      1. `random`
      2. `greedy`
3. **Execution** : Run the script using the appropriate Python command. For example:

 For example:

```python
   python run_local_search.py --dataset voc --dn_type lr --search_method greedy
```

1. **Results** : The script will execute the Local Search-Based Methods and provide results, which can be analyzed for model performance improvements.

For more detailed instructions and additional options, please refer to the documentation provided within the `run_local_search.py` script.

### 2. Multi-Linear Integer Programming

Our second inference scheme is based on Multi-Linear Integer Programming. This mathematical optimization approach is tailored to finding the MPE value.

#### Usage

To utilize the Multi-Linear Integer Programming-based inference scheme, follow these steps:

1. **Navigate to the Directory**: Locate the `run_ilp.py` script within this directory.
2. **Configuration**: Modify the script's configuration file to specify any necessary parameters, such as dataset selection and other options.

   `--dataset`: Name of the dataset you want to use (default is "nus").
   `--dn_type`: Name of the Dependency Network type (default is "lr").
   `--debug`: Use this flag if you want to enable debug mode.
   `--batch_size`: Specify the batch size (default is 64).
3. Adjust these parameters to match your experiment's specific settings.
4. **Execution**: Run the script using the appropriate Python command as per your configuration. For example:

   ```bash
      python run_ilp.py --dataset your_dataset_name --dn_type lr --debug --batch_size 128
   ```


### Updating `base_path` in `utils.py`

For all the inference methods provided in this repository, it's essential to configure the `base_path` in the `utils.py` file correctly. The `base_path` specifies the directory where the output of the feature extractor and the trained dependency network should be located. Follow these steps to update the `base_path`:

1. **Locate the `utils.py` File**: Navigate to the directory containing your Python scripts, and find the `utils.py` file.
2. **Edit the `base_path` Variable**: Open the `utils.py` file and locate the `get_cnn_output_and_model(dataset, method)` function. In this function, you'll find a variable named `base_path`.
3. **Set `base_path` to the Appropriate Directory**: Set the `base_path` variable to the directory where the output of the feature extractor and the trained dependency network are stored. Ensure that this directory matches the paths specified within the `get_cnn_output_and_model(dataset, method)` function.

Here's an example of how to update the `base_path` variable:

```python
# Inside the utils.py file
def get_cnn_output_and_model(dataset, method):
    # Define the base path here
    base_path = "/path/to/your/directory"

    # Rest of the function code
```
