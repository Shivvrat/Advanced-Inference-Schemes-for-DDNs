import time
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import numpy as np


from gurobipy import Model, GRB, quicksum


def compute_mpe_using_gurobi(args):
    """
    nn_dict: Dictionary containing weights and biases for each layer
    input_values: Specific input values to compute the output for

    Returns the computed output for the given input values
    """
    nns_dict, input_values, target, idx = args

    # Suppress the Gurobi output
    if not debug:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            model = Model("DN-MPE-ILP", env=env)
            model.setParam("TimeLimit", 60 * 60)  # 1 hour
    else:
        model = Model("DN-MPE-ILP")

    # model.setParam("SolutionLimit", 100)
    cnn_feature_size = len(input_values)
    model._cur_obj = float("inf")
    model._time = time.time()

    # Define the inputs to the models - each model will take (2n-1) inputs.
    # Thus we need to remove the value of y for that index
    cnn_features_vars = model.addVars(
        cnn_feature_size, lb=0.0, ub=1.0, name="cnn_features"
    )
    # mpe_vars = model.addVars(cnn_feature_size, lb=0.0, ub=1.0, name="mpe_output")
    mpe_vars = model.addVars(cnn_feature_size, vtype=gp.GRB.BINARY, name="mpe_output")

    objectives = []
    for idx in range(cnn_feature_size):
        model.addConstr(cnn_features_vars[idx] == input_values[idx])

    for nn_idx, this_nn_dict in nns_dict.items():
        # Defining input layer variables with bounds
        input_to_model = model.addVars(
            2 * cnn_feature_size - 1, lb=-float("inf"), name=f"input_{nn_idx}"
        )

        for ip_idx in range(2 * cnn_feature_size):
            if ip_idx == nn_idx:
                continue
            use_idx = ip_idx - 1 if ip_idx > nn_idx else ip_idx
            if ip_idx < cnn_feature_size:
                model.addConstr(input_to_model[use_idx] == mpe_vars[ip_idx])
            else:
                index_for_cnn_features = use_idx - cnn_feature_size + 1
                model.addConstr(
                    input_to_model[use_idx] == cnn_features_vars[index_for_cnn_features]
                )

        # For each layer in the neural network
        output_before_activation = model.addVar(
            lb=-float("inf"), name=f"output_before_activation_{nn_idx}"
        )
        prev_layer_vars = input_to_model
        for i, layer_idx in enumerate(this_nn_dict):
            weights = this_nn_dict[layer_idx]["weight"]
            biases = this_nn_dict[layer_idx]["bias"]

            # Define the variables for this layer
            layer_size = len(biases)
            layer = model.addVars(layer_size, name=f"layer_{i}")

            for j in range(layer_size):
                lin_expr = (
                    quicksum(
                        weights[j, k] * prev_layer_vars[k]
                        for k in range(len(prev_layer_vars))
                    )
                    + biases[j]
                )
                lin_expr_var = model.addVar(
                    name=f"lin_expr_{nn_idx}_{i}_{j}", lb=-float("inf")
                )
                model.addConstr(lin_expr_var == lin_expr)

                # For ReLU activation
                if i < len(this_nn_dict) - 1:
                    relu_out = model.addVar(name=f"relu_out_{nn_idx}_{i}_{j}")
                    model.addGenConstrMax(
                        relu_out, [0, lin_expr_var], name=f"ReLU_{nn_idx}_{i}_{j}"
                    )
                    model.addConstr(layer[j] == relu_out)
                else:
                    model.addConstr(output_before_activation == lin_expr_var)

            prev_layer_vars = layer

        # e^z
        output_exp = model.addVar()
        model.addGenConstrExp(
            output_before_activation,
            output_exp,
            options="FuncPieces=-1 FuncPieceError=0.01",
        )

        # Now the expression 1+e^z
        value_rhs = model.addVar()
        model.addConstr(value_rhs == 1 + output_exp)

        # Now the expression log_e(1+e^z)
        log_value_rhs = model.addVar(lb=-float("inf"))
        model.addGenConstrLog(
            value_rhs, log_value_rhs, options="FuncPieces=-1 FuncPieceError=0.01"
        )

        # Now the expression xz (need to add x)
        value_l = model.addVar(lb=-float("inf"))
        model.addConstr(value_l == output_before_activation * mpe_vars[nn_idx])
        objectives.append(value_l - log_value_rhs)

    model.setObjective(quicksum(objectives), GRB.MAXIMIZE)

    # Extract and return the computed output for the given input values
    # model.optimize(callback=cb)
    model.optimize()
    # Check the optimization status
    if model.status != GRB.OPTIMAL:
        raise Exception(f"Gurobi optimization failed. Status code: {model.status}")

    vals_y = [mpe_vars[var_idx].X for var_idx in mpe_vars]
    return idx, vals_y, target


# Example
import numpy as np


def simple_nn(input_values, nn_dict):
    layer_input = input_values
    for i in range(len(nn_dict)):
        if i == len(nn_dict) - 1:
            # use sigmoid activation layer
            activation_layer = lambda x: 1 / (1 + np.exp(-x))
        else:
            # use ReLU activation layer
            activation_layer = lambda x: np.maximum(0, x)
        z = np.dot(nn_dict[i]["weight"], layer_input) + nn_dict[i]["bias"]
        layer_input = activation_layer(z)
    return layer_input, z


def generate_nn(num_inputs, num_outputs):
    # Randomly generate weights and biases based on provided dimensions
    return {
        0: {  # First layer (hidden layer)
            "weight": np.random.uniform(
                -2, 2, (2, num_inputs)
            ),  # 2 neurons in the hidden layer
            "bias": np.random.uniform(-2, 2, 2),
        },
        1: {  # Second layer (output layer)
            "weight": np.random.uniform(
                -2, 2, (num_outputs, 2)
            ),  # 2 neurons from the hidden layer
            "bias": np.random.uniform(-2, 2, num_outputs),
        },
    }


def cb(model, where):
    if where == GRB.Callback.MIPNODE:
        # Get model objective
        obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

        # Has objective changed?
        if abs(obj - model._cur_obj) > 1e-8:
            # If so, update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # Terminate if objective has not improved in 20s
    if time.time() - model._time > 20:
        model.terminate()


debug = False

if __name__ == "__main__":
    num_inputs = 3
    num_outputs = 1

    # test_inputs = [[0.7, 0.2], [0.2, 0.8], [0.8, 0.2], [0.3, 0.3], [0.7, 0.7]]
    cnn_predictions = np.random.rand(100, num_inputs)
    # test_inputs = np.random.randint(0, 5, size=(10, 2))
    for cnn_features in cnn_predictions:
        nns_dict = {
            i: generate_nn(2 * num_inputs - 1, num_outputs) for i in range(num_inputs)
        }
        args = (nns_dict, cnn_features, None, None)

        idx, vaLs_y, target = compute_mpe_using_gurobi(args)
        print(f"Computed value of - {vaLs_y} for input {cnn_features}")
