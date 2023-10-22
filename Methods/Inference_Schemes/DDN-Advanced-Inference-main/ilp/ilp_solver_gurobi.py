import sys

import numpy as np

# import pyscipopt
from loguru import logger
from numpy import argmax

# from pyscipopt import Model, quicksum, quickprod
import gurobipy as gp


class CheckILPOutput:
    def __init__(self, name, args, nn, input_to_nn, debug=False):
        self.weights = None
        self.bias = None
        self.nn = nn
        self.variables = {}
        self.objective_values = []
        self.solver = gp.Model(name)
        if not debug:
            self.solver.Params.TimeLimit = 120
        else:
            self.solver.Params.TimeLimit = 60

        self.num_layers = len(nn)
        # Layer number starts at zero - Values at zero is that of input
        self.input_as_vector = input_to_nn
        assert (
            self.input_as_vector.shape[0] == args.input_size
        ), "Input vector to ILP is not correct"
        for idx, val in enumerate(self.input_as_vector):
            self.variables[0, idx, "x"] = self.solver.addVar(
                name=f"modified_input_idx_{idx}_x", lb=0, ub=1, vtype="C"
            )
            if args.func_f == "min_distance":
                self.add_constraints_for_f_min_distance(idx)

        for idx, val in enumerate(self.input_as_vector):
            if args.func_f == "min_distance_plus_grid":
                self.add_constraints_for_f_min_distance_plus_grid(idx)

    # @njit()
    def add_constraints_one_layer(self, layer_number, last_layer=False):
        # the size of weight vector is [output, input]
        for idx_o in range(self.weights.shape[0]):
            # The variables with key z are the values before activation function
            # Lower bound for z needs to be none since it can be negative as well!
            # The variables with key 0 are the values after activation function is applied
            if not last_layer:
                self.variables[layer_number, idx_o, "x"] = self.solver.addVar(
                    name=f"op_{layer_number}_idx_{idx_o}_x",
                    lb=0,
                    vtype=gp.GRB.CONTINUOUS,
                )
            else:
                self.variables[layer_number, idx_o, "x"] = self.solver.addVar(
                    name=f"op_{layer_number}_idx_{idx_o}_x",
                    lb=None,
                    vtype=gp.GRB.CONTINUOUS,
                )
            # if last_layer:
            self.variables[layer_number, idx_o, "s"] = self.solver.addVar(
                name=f"op_{layer_number}_idx_{idx_o}_s", lb=0
            )
            # self.variables[layer_number, idx_o, "z"] = self.model.addVar(name=f"op_{layer_number}_idx_{idx_o}_s",
            #                                                              lb=0, ub=1, vtype="I",)
            # Add the values of all output from ReLu as object function
            if not last_layer:
                self.objective_values.append(self.variables[layer_number, idx_o, "x"])
                # No need for z since SCIP takes care of the indicator constraints
                # self.objective_values.append(self.variables[layer_number, idx_o, "z"])
            this_sum = []
            # Calculate output for layer "layer_number" and node index "idx_o"
            for idx_i in range(self.weights.shape[1]):
                # Values of w*x
                this_sum.append(
                    self.variables[layer_number - 1, idx_i, "x"]
                    * self.weights[idx_o, idx_i]
                )
            # Values of b
            this_sum.append(self.bias[idx_o])
            if not last_layer:
                # Constraint -> x-s = sum_i(w_i*x_i)
                self.solver.addCons(
                    sum(this_sum)
                    == (
                        self.variables[layer_number, idx_o, "x"]
                        - self.variables[layer_number, idx_o, "s"]
                    ),
                )
                # Only one of x or s can be > 0
                self.solver.addConsCardinality(
                    [
                        self.variables[layer_number, idx_o, "x"],
                        self.variables[layer_number, idx_o, "s"],
                    ],
                    1,
                )
            else:
                # For final layer we just need the sum, since we apply softmax
                self.solver.addCons(
                    sum(this_sum) == self.variables[layer_number, idx_o, "x"]
                )

    def add_constraint_all_layers(
        self,
    ):
        # num_layers = self.num_layers
        # self.weights = self.nn[f"l1.weight"]
        # self.bias = self.nn[f"l1.bias"]
        # self.add_constraints_one_layer(0 + 1, True)

        for each_layer in range(0, self.num_layers):
            last_layer = False
            self.weights = self.nn[each_layer]["weight"]
            self.bias = self.nn[each_layer]["bias"]
            if each_layer == self.num_layers - 1:
                last_layer = True
            self.add_constraints_one_layer(each_layer + 1, last_layer)


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def create_solver(args):
    """
    Create a solver for the ILP task
    :param name:
    :param nn:
    :param input_to_nn:
    :return:
    """
    name, nn, output_of_nn, target, debug, main_args, index = args
    # Define a dictionary to hold the variables in the ILP problem
    output_of_nn = output_of_nn.reshape(-1)
    ilp_model = CheckILPOutput("ILP_for_nn", main_args, nn, output_of_nn, debug)
    if debug:
        ilp_model.solver.hideOutput(False)
    else:
        ilp_model.solver.hideOutput(True)

    ilp_model.add_constraint_all_layers()
    objective = sum(ilp_model.objective_values)
    if target == 1:
        ilp_model.solver.addCons(
            ilp_model.variables[ilp_model.num_layers, 0, "x"] <= -1e-10
        )
    else:
        ilp_model.solver.addCons(
            -ilp_model.variables[ilp_model.num_layers, 0, "x"] <= 0
        )
    ilp_model.solver.setObjective(objective, "minimize")
    ilp_model.solver.optimize()
    if ilp_model.solver.getStatus() != "optimal":
        logger.info(
            f"Didn't find optimal solution for example index {index} due to time limit"
        )
        return index, None
    updated_input = np.zeros_like(output_of_nn)
    for v in ilp_model.solver.getVars():
        name, value = v.name, ilp_model.solver.getVal(v)
        if name.startswith(f"modified_input_idx_") and name[-1] == "x":
            input_index = int(name.split("_")[-2])
            updated_input[input_index] = float(value)
        elif name.startswith(f"op_{ilp_model.num_layers}_") and name[-1] == "x":
            output_index = int(name.split("_")[-2])
            final_output = float(value)
    if final_output > 0:
        final_output = 1
    else:
        final_output = 0
    if debug:
        logger.info("We flipped the class!")
        logger.info(f"Output of nn {final_output}")
        logger.info(f"True Label {target}")
    if updated_input.shape[0] == 0:
        return index, None
    else:
        return index, updated_input
