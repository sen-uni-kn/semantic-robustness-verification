# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import logging
import numpy as np
import torch
from pyscipopt import Model, quicksum
import bounds as bounds
from distance import Distance
from tqdm import tqdm
import time

def translate_network(model: Model, torch_model, input_dim, relu_bounds, output_bounds):
    """
    Translate a PyTorch sequential feedforward network into a SCIP model using MILP.

    Args:
        model (Model): PySCIPOpt model to which variables and constraints will be added.
        torch_model (torch.nn.Module): PyTorch sequential model.
        input_dim (tuple): Input dimensions.
        relu_bounds (list): List of tuples of lower and upper bounds for ReLU nodes.
        output_bounds (tuple): Lower and upper bounds for output layer.

    Returns:
        input_vars (list): List of SCIP input variables.
        output_vars (list): List of SCIP output variables representing network output.
    """

    def add_relu_constraint(model, z, lb, ub, idx, neuron_idx):
        # Create output variable for ReLU (after activation)
        y = model.addVar(name=f"relu_output_{idx+1}_{neuron_idx}", lb=0, ub=max(0, ub))

        if lb >= 0: # stably active
            model.addCons(y == z)  
        elif ub <= 0: # stably inactive
            model.addCons(y == 0)  
        else: # unstable
            a = model.addVar(vtype="B", name=f"relu_activation_{idx+1}_{neuron_idx}") 
            
            model.addCons(y <= z - lb * (1 - a))  # y ≤ z − lb(1 − a)
            model.addCons(y >= z)                 # y ≥ z
            model.addCons(y <= ub * a)            # y ≤ ub * a
            model.addCons(y >= 0)                 # y ≥ 0

        return y

    layer_weights = []
    layer_biases = []
    for layer in torch_model.sequential:
        if isinstance(layer, torch.nn.Linear):
            layer_weights.append(layer.weight.detach().numpy())
            layer_biases.append(layer.bias.detach().numpy())

    params = []
    for idx, (w, b) in enumerate(zip(layer_weights, layer_biases)):
        params.append((w, b))

    # Create a SCIP variable array for the input to the neural network
    input_vars = [model.addVar(lb=0, ub=1, name=f"input_var_{i}") for i in range(int(np.prod(input_dim)))]

    def forward(input_vars, params, relu_bounds, output_bounds):
        x = input_vars

        # Iterate over each hidden layer's weights and biases, except for the last one
        for idx, (w, b) in enumerate(params[:-1]):
            hidden_output = []

            for i in range(w.shape[0]):
                
                z = quicksum(w[i][j] * x[j] for j in range(len(x))) + b[i]              # Add constraint for the linear operation: w * x + b
                lb_i, ub_i = relu_bounds[idx][0][i], relu_bounds[idx][1][i]             # Get bounds for current ReLU node
                hidden_output.append(add_relu_constraint(model, z, lb_i, ub_i, idx, i)) # Add ReLU constraints using the submethod

            # After activation, set x = hidden_output for the next layer
            x = hidden_output

        # Handle the final layer (output layer) without ReLU activation
        w_last, b_last = params[-1]
        l_bnds, u_bnds = output_bounds

        #network_output = [model.addVar(lb=l_bnds[i], ub=u_bnds[i], name=f"output_var_{i}") for i in range(w_last.shape[0])]
        network_output = [model.addVar(lb=l_bnds[i], ub=u_bnds[i], name=f"output_var_{i}") for i in range(w_last.shape[0])]
        for i in range(w_last.shape[0]):
            model.addCons(quicksum(w_last[i][j] * x[j] for j in range(len(x))) + b_last[i] == network_output[i])

        return network_output

    return input_vars, forward(input_vars, params, relu_bounds, output_bounds)

# Encode max-constraints in PySCIPOpt
def add_max_constraint(scip_model, x, output_bounds):
    """
    Adds a one-hot encoded max constraint to identify the maximum among output variables.

    Args:
        scip_model (Model): PySCIPOpt model.
        x (list): Output variables of the neural network.
        output_bounds (tuple): Lower and upper bounds for the output variables.

    Returns:
        a (list): List of binary indicator variables.
        y (SCIP variable): Variable representing the maximum output.
    """
        
    l_bnds, u_bnds = output_bounds

    y = scip_model.addVar(name="max_out", lb=min(l_bnds), ub=max(u_bnds))
    a = [scip_model.addVar(vtype="B", name=f"a_{i}") for i in range(len(x))]

    for i in range(len(x)):
        scip_model.addCons(y <= x[i] + (1 - a[i]) * (max(u_bnds) - l_bnds[i]))
        scip_model.addCons(y >= x[i])

    scip_model.addCons(quicksum(a[i] for i in range(len(a))) == 1)
    return a, y
    
# Convert list of PySCIPOpt variables into array
def list_to_value(scip_model, scip_variables):
    """
    Converts a list of SCIP variables into their evaluated float values.

    Args:
        scip_model (Model): The SCIP model from which to read variable values.
        scip_variables (list): List of SCIP variables.

    Returns:
        list: List of variable values.
    """
    return [scip_model.getVal(var) for var in scip_variables]

# Get variable by name from SCIP model
def get_var_by_name(scip_model, var_name):
    """
    Retrieves a SCIP variable from the model by name.

    Args:
        scip_model (Model): PySCIPOpt model.
        var_name (str): Name of the variable.

    Returns:
        variable: SCIP variable.

    Raises:
        ValueError: If variable is not found.
    """
    for var in scip_model.getVars():
        if var.name == var_name:
            return var
    raise ValueError(f"Variable {var_name} not found in SCIP model.")

def solve(params, scip_model, concurrent):
    """
    Solves the SCIP model with optional concurrency and custom parameters.

    Args:
        params: Parameter object containing solving configuration.
        scip_model (Model): The SCIP model to solve.
        concurrent (bool): Whether to use concurrent solving.
    """

    scip_model.setParam("limits/solutions", 1)
    scip_model.setParam("presolving/maxrounds", 0)
    scip_model.setParam("display/verblevel", 5)
    scip_model.setParam("numerics/feastol", 1e-10) 
    scip_model.setParam("numerics/dualfeastol", 1e-10)  

    scip_model.setParam("randomization/lpseed", 42)
    scip_model.setParam("randomization/permutationseed", 42)
    scip_model.setParam("branching/random/seed", 42)
    scip_model.setParam("branching/relpscost/startrandseed", 42)
    scip_model.setParam("heuristics/alns/seed", 42)
    scip_model.setParam("separating/zerohalf/initseed", 42)
    scip_model.setParam("randomization/randomseedshift", 0)

    if params.minlp.timeout > 0:
        scip_model.setParam("limits/time", params.minlp.timeout)

    scip_model.hideOutput(False)

    if concurrent:
        logging.info("run scip optimisation concurrent ...")
        scip_model.setParam("parallel/minnthreads", 12)
        scip_model.setParam("parallel/maxnthreads", 24)
        scip_model.solveConcurrent()
    else:
        logging.info("run scip optimisation single threaded ...")
        scip_model.optimize()

def compare_scip_torch_outputs(params, scip_model, torch_model, input_dim, output_dim, iterations, eps=None):
    """
    Compares the output of the SCIP-encoded model to the PyTorch model on random inputs.

    Args:
        params: Parameters including tolerance value.
        scip_model (Model): SCIP model of the network.
        torch_model (torch.nn.Module): PyTorch model of the network.
        input_dim (int): Input dimension size.
        output_dim (int): Output dimension size.
        iterations (int): Number of samples to test.
        eps (float, optional): Error tolerance for output deviation.

    Returns:
        float: Maximum deviation observed.
    """

    max_deviation = 0.0
    with tqdm(total=iterations, desc="max deviation: 0.0e-00", dynamic_ncols=True) as pbar:
        for _ in range(iterations):
            random_input = [np.random.uniform(low=0, high=1) for _ in range(input_dim)]

            torch_input = torch.tensor(random_input, dtype=torch.float32).unsqueeze(0)
            torch_model.eval()
            with torch.no_grad():
                torch_output = torch_model(torch_input).squeeze(0).detach().numpy()

            scip_model_copy = Model(sourceModel=scip_model)
            for i in range(input_dim):
                scip_var = get_var_by_name(scip_model_copy, f"input_var_{i}")
                scip_model_copy.chgVarLb(scip_var, random_input[i])
                scip_model_copy.chgVarUb(scip_var, random_input[i])

            scip_model_copy.freeTransform()
            scip_model_copy.hideOutput()
            scip_model_copy.optimize()

            if scip_model_copy.getStatus() not in ['optimal', 'feasible']:
                logging.error(f"forward optimization status {scip_model_copy.getStatus()}")
                raise Exception(f"scip network model test failed!")

            scip_output = [scip_model_copy.getVal(get_var_by_name(scip_model_copy, f"output_var_{i}")) for i in range(output_dim)]
            max_out = scip_model_copy.getVal(get_var_by_name(scip_model_copy, "max_out"))
            a_values = [scip_model_copy.getVal(get_var_by_name(scip_model_copy, f"a_{i}")) for i in range(output_dim)]

            deviation = np.abs(torch_output - np.array(scip_output))
            max_sample_deviation = np.max(deviation)
            max_deviation = max(max_deviation, max_sample_deviation)

            if not np.isclose(np.max(torch_output), max_out, atol=eps):
                raise Exception(f"scip network model test failed, max_out mismatch. \n torch: {torch_output}, scip: {scip_output}, max scip: {max_out}, one hot scip: {a_values}")

            pbar.set_description(f"max deviation: {max_deviation:.1e}")
            pbar.update(1)

            if eps is not None and max_sample_deviation > eps:
                pbar.update(pbar.total - pbar.n)
                logging.error(f"scip network model test failed, deviation exceeds tolerance: {max_sample_deviation:.1e} > {eps}")
                return

            scip_model_copy.freeProb()
    logging.info(f"maximum model output deviation: {max_deviation:.1e}")
    return max_deviation

def compute_bounds_for_output(scip_model, output_dim):
    """
    Computes tight lower and upper bounds on the output of the neural network.

    Args:
        scip_model (Model): The SCIP model.
        output_dim (int): Dimension of the output layer.

    Returns:
        tuple: Lower and upper bounds (arrays) for the output variables.
    """
    l_bnds = np.zeros(output_dim)
    u_bnds = np.zeros(output_dim)

    scip_model_copy = Model(sourceModel=scip_model)

    for j in range(output_dim):
        output_var_name = f"output_var_{j}"
        scip_output_var = get_var_by_name(scip_model_copy, output_var_name)
        
        # Minimization
        scip_model_copy.freeTransform() 
        scip_model_copy.setObjective(scip_output_var, sense='minimize')
        #scip_model_copy.hideOutput()
        scip_model_copy.optimize()
        l_bnds[j] = scip_model_copy.getVal(scip_output_var)

        # Maximization
        scip_model_copy.freeTransform() 
        scip_model_copy.setObjective(scip_output_var, sense='maximize')
        #scip_model_copy.hideOutput()
        scip_model_copy.optimize()
        u_bnds[j] = scip_model_copy.getVal(scip_output_var)

    # Free the model
    scip_model_copy.freeProb()

    return l_bnds, u_bnds

# creates a milp program for the given network and property and runs the optimiser
def verify(params, torch_model, ri, ev_verify):
    """
    Encodes a robustness verification task as a MILP problem and runs the verifier.

    Args:
        params: Configuration and hyperparameters for verification.
        torch_model (torch.nn.Module): The PyTorch neural network.
        ri (torch.Tensor): Reference input to be perturbed.
        ev_verify (dict): Dictionary to update with verification metrics.

    Returns:
        bool or list: True if verified, False if time limit exceeded, or counterexample input as list.
    """
    distance    = getattr(Distance, params.minlp.distance)
    threshold   = params.minlp.threshold

    tolerance_e = (10 ** (-params.minlp.tolerance))

    relu_bounds, output_bounds = bounds.calculate_bounds(torch_model, params.dimension, 0)

    stable_relu = len(list(filter(
        lambda entry: any(lb >= 0 for lb in entry[0]) or any(ub <= 0 for ub in entry[1]),
            relu_bounds
        )))

    scip_model = Model()
    scip_input_vars, scip_output_vars = translate_network(model=scip_model, 
                                                          torch_model=torch_model, 
                                                          input_dim=params.dimension, 
                                                          relu_bounds=relu_bounds, 
                                                          output_bounds=output_bounds)

    scip_onehot, scip_maxout = add_max_constraint(scip_model, scip_output_vars, output_bounds)
    scip_distance = distance(scip_input_vars, ri.tolist(), scip_model, split=True)

    ev_verify.update({"max_deviation": compare_scip_torch_outputs(params, scip_model, torch_model, len(scip_input_vars), len(scip_output_vars), 1000, tolerance_e)})
    
    scip_model.addCons(scip_onehot[params.ri.class_idx] == 0)
    scip_model.addCons(scip_distance >= threshold)
    
    scip_model.setObjective(scip_distance, sense="maximize")

    start_time = time.perf_counter()
    solve(params, scip_model, False)
    end_time = time.perf_counter()

    ev_verify.update({"stable_relu": stable_relu})
    ev_verify.update({"runtime": end_time - start_time})
    ev_verify.update({"status": scip_model.getStatus()})

    if scip_model.getStatus() in ['infeasible']:
        logging.info(f'model is infeasible, no further counterexamples found, verification runtime: {end_time - start_time:.4f}s')
        return True

    # check if model found an optimal or feasible solution
    elif scip_model.getStatus() in ['optimal', 'feasible', 'sollimit']:

        input_vars = list_to_value(scip_model, scip_input_vars)
        output_vars = list_to_value(scip_model, scip_output_vars)
        onehot = list_to_value(scip_model, scip_onehot)
        distance_value = scip_model.getVal(scip_distance)
        maxout = scip_model.getVal(scip_maxout)
        distance_torch = distance(torch.tensor(input_vars), ri)

        torch_model.eval()
        with torch.no_grad():
            torch_output = torch_model(torch.tensor(input_vars, dtype=torch.float32).unsqueeze(0)).squeeze(0)

        torch_output_np = torch_output.detach().numpy()
        abs_error = max(np.abs(torch_output_np - output_vars))

        # Assertion 1
        assert np.max(abs_error) <= tolerance_e, f"Assertion failed: np.max(abs_error) = {np.max(abs_error)} > tolerance_e = {tolerance_e}"

        # Assertion 2
        rounded_distance_value = np.round(distance_value, params.minlp.tolerance)
        rounded_distance_torch = np.round(distance_torch, params.minlp.tolerance)
        if rounded_distance_value != rounded_distance_torch:
            logging.warning(f"Assertion failed: Rounded distance_value = {rounded_distance_value} != Rounded distance_torch = {rounded_distance_torch} with tolerance = {params.minlp.tolerance}")

        # Assertion 3
        assert rounded_distance_value >= params.minlp.threshold, (
            f"Assertion failed: Rounded distance_value = {rounded_distance_value} < "
            f"Threshold = {params.minlp.threshold}"
        )

        if np.argmax(output_vars) == params.ri.class_idx:
            arr_copy = np.delete(output_vars, np.argmax(output_vars))
            assert np.max(output_vars) <= np.max(arr_copy) + tolerance_e , f"assertion failed, {np.max(output_vars)} is not <= {np.max(arr_copy)} + {tolerance_e} \noutput vars {output_vars}, arr copy {arr_copy}"

        return input_vars
        
    elif scip_model.getStatus() in ['userinterrupt']:
        logging.info(f'interrupted by user')
        exit()
    elif scip_model.getStatus() in ['timelimit']:
        return False
    else:
        raise Exception(f"unknown return code: {scip_model.getStatus()}")
