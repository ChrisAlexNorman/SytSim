import numpy as np
from scipy.integrate import cumtrapz
from utilities import sparsify_rate_matrix, eval_params_in_array

### CLAMP MODEL BUILDER ###

def clamp_model_builder(architecture, fusion_model='clamp_exponential', replenishment_model=None):
    """Build sparse stochastic version. Markov model from specifications for a SV fusion model
    
    Inputs
    ------
    architecture : list of lists of dicts
        For each clamp a list of model specifications for each sensor.
    fusion_model : str. One of {'clamp_exponential', 'clamp_exponential_with_resistance'}
        Specification for which fusion model to use.
    replenishment_model : str or None. Default: None.
        Specification for how vesicle is replenished after fusion.

    Returns
    -------
    model : dict
    """

    state_names = []
    initial_condition_names = []
    transitions = {}

    # Loop over clamps
    for i_clamp, sensors in enumerate(architecture):
        # For each clamp, append each sensor's properties to the model
        for i_sensor, sensor_model in enumerate(sensors):

            # Append state names
            sensor_state_names = [f"{name}{i_clamp}{i_sensor}" for name in sensor_model["state_names"]]
            state_names.extend(sensor_state_names)

            # Append initial conditions 
            for index, condition in enumerate(sensor_model["initial_condition"]):
                if condition: initial_condition_names.append(sensor_state_names[index])

            # Append non-zero transitions
            rate_matrix = eval_params_in_array(sensor_model["rate_matrix"], sensor_model["parameters"])
            transitions = sparsify_rate_matrix(rate_matrix, sensor_state_names, transitions)
    
    # Add clamp states
    for i_clamp in range(len(architecture)+1):
        state_names.append(f"{i_clamp}_free_clamps")
    initial_condition_names.append("0_free_clamps")

    # Add fusion states
    state_names.extend(["Unfused", "Fused"])
    initial_condition_names.append("Unfused")
    # Add fusion rate
    transitions["Unfused"] = [{
        'destination': 'Fused',
        'rate': get_fusion_rate(fusion_model)
    }]
    # Add replenishment rate
    transitions["Fused"] = [{
        'destination': 'Unfused',
        'rate': get_replenishment_rate(replenishment_model)
    }]

    if fusion_model == 'clamp_exponential':
        rate_update_rule = clamp_model_rate_updates
    elif fusion_model == 'clamp_exponential_with_resistance':
        rate_update_rule = clamp_model_rate_updates_with_resistance

    # return collection
    model = {
        'transitions' : transitions,
        'initial_condition_names': initial_condition_names,
        'state_names': state_names,
        'state_update_rule': clamp_model_state_updates,
        'rate_update_rule': rate_update_rule
    }
    return model

### FUSION & REPLENISHMENT RATES ###

def get_fusion_rate(model_name):
    if model_name is None:
        model_name = 'none'
    model_name = model_name.lower()
    
    recognised_models = ['none', 'constant', 'clamp_exponential', 'clamp_exponential_with_resistance']
    if model_name not in recognised_models:
        raise ValueError(f"Fusion model '{model_name}' not recognised. Must be one of {recognised_models}.")
    
    if model_name == 'none':
        return 0
    
    if model_name == 'constant':
        return 1

    if model_name == 'clamp_exponential':
        return 1
    
    if model_name == 'clamp_exponential_with_resistance':
        return 1

def get_replenishment_rate(model_name):
    if model_name is None:
        model_name = 'none'
    model_name = model_name.lower()

    recognised_models = ['none', 'constant']
    if model_name not in recognised_models:
        raise ValueError(f"Replenishment model '{model_name}' not recognised. Must be one of {recognised_models}.")
    
    if model_name == 'none':
        return 0
    
    if model_name == 'constant':
        return 0.02

### STATE UPDATE FUNCTIONS ###

def clamp_model_state_updates(current_state, simulation, chosen_transition=None, changed_state=None):
    if changed_state is None:
        changed_state = []
    # Apply selected transition
    current_state, changed_state = apply_transition(current_state, simulation, chosen_transition, changed_state)
    # Re-initialise state if vesicle was replenished.
    current_state, changed_state = reinitialise_vesicle(current_state, simulation, chosen_transition, changed_state)
    # Update free clamps
    current_state, changed_state = update_free_clamps(current_state, simulation, chosen_transition, changed_state)

    return current_state, changed_state

def apply_transition(current_state, simulation, chosen_transition=None, changed_state=None):
    if changed_state is None:
        changed_state = []
    for index, state in enumerate(current_state):
        if state == chosen_transition['source']:
            current_state[index] = chosen_transition['destination']
            changed_state.append(chosen_transition['destination'])
    return current_state, changed_state

def update_free_clamps(current_state, simulation, chosen_transition=None, changed_state=None):
    if changed_state is None:
        changed_state = []
    # Count number of free clamps
    _, n_free = count_clamps(current_state)

    # Update current state if changed
    free_clamp_state_name = f"{n_free}_free_clamps"
    if free_clamp_state_name not in current_state:
        # Remove existing free_clamp state
        i_clamp = 0
        while True:
            test_state_name = f"{i_clamp}_free_clamps"
            if test_state_name not in simulation["state_names"]:
                break
            if test_state_name in current_state:
                current_state.remove(test_state_name)
            i_clamp += 1
        # Add new free_clamp state
        current_state.append(free_clamp_state_name)
        changed_state.append(free_clamp_state_name)
    
    return current_state, changed_state

def reinitialise_vesicle(current_state, simulation, chosen_transition=None, changed_state=None):
    if changed_state is None:
        changed_state = []
    if chosen_transition["source"] == "Fused" and chosen_transition["destination"] == "Unfused":
        current_state = changed_state = simulation["initial_condition"]
    return current_state, changed_state

### RATE UPDATE FUNCTIONS ###

def clamp_model_rate_updates_with_resistance(current_state, simulation, chosen_transition=None, current_transitions=None):
    if current_transitions is None:
        current_transitions = []
    # Collect all transitions possible from current state
    current_transitions = get_feasible_transitions(current_state, simulation, chosen_transition, current_transitions)
    # If vesicle fused, make exiting the fused state the only viable transition
    current_transitions = fusion_cancels_other_transitions(current_state, simulation, chosen_transition, current_transitions)
    # Update the rate of fusion according to Arrhenius equation for how many clamps are free
    current_transitions = exponential_clamp_fusion_rate_with_resistance(current_state, simulation, chosen_transition, current_transitions)

    return current_transitions

def clamp_model_rate_updates(current_state, simulation, chosen_transition=None, current_transitions=None):
    if current_transitions is None:
        current_transitions = []
    # Collect all transitions possible from current state
    current_transitions = get_feasible_transitions(current_state, simulation, chosen_transition, current_transitions)
    # If vesicle fused, make exiting the fused state the only viable transition
    current_transitions = fusion_cancels_other_transitions(current_state, simulation, chosen_transition, current_transitions)
    # Update the rate of fusion according to Arrhenius equation for how many clamps are free
    current_transitions = exponential_clamp_fusion_rate(current_state, simulation, chosen_transition, current_transitions)

    return current_transitions

def fusion_cancels_other_transitions(current_state, simulation, chosen_transition=None, current_transitions=None):
    if current_transitions is None:
        current_transitions = []
    # If vesicle fused, make exiting the fused state the only viable transition
    if chosen_transition is not None:
        if chosen_transition["source"] == "Unfused" and chosen_transition["destination"] == "Fused":
            current_transitions = []
            for transition in simulation["transitions"]["Fused"]:
                current_transitions.append(transition|{"source": "Fused"})
            # current_transitions = simulation["transitions"]["Fused"] # [transitions for source, transitions in simulation["transitions"].items() if source == 'Fused']
    return current_transitions

def exponential_clamp_fusion_rate(current_state, simulation, chosen_transition=None, current_transitions=None):
    if current_transitions is None:
        current_transitions = []
    # Update the rate of fusion according to Arrhenius equation for how many clamps are free
    _, n_free = count_clamps(current_state)
    fusion_rate = np.exp(21.5) * 1e-3 * np.exp(n_free * 4.5 - 26)
    for transition in current_transitions:
        if transition["source"] == "Unfused" and transition["destination"] == "Fused":
            transition["rate"] = np.repeat(fusion_rate, 2)
            transition["rate_integral"] = cumtrapz(transition["rate"], transition["timestamp"], initial=0)
    return current_transitions

def exponential_clamp_fusion_rate_with_resistance(current_state, simulation, chosen_transition=None, current_transitions=None):
    if current_transitions is None:
        current_transitions = []
    # Update the rate of fusion according to Arrhenius equation for how many clamps are free
    n_clamps, n_free = count_clamps(current_state)
    k = 1
    fusion_rate = np.exp(21.5) * 1e-3 * np.exp(n_free * (4.5 + k) - 26 - k*n_clamps)
    for transition in current_transitions:
        if transition["source"] == "Unfused" and transition["destination"] == "Fused":
            transition["rate"] = np.repeat(fusion_rate, 2)
    return current_transitions

def get_feasible_transitions(current_state, simulation, chosen_transition=None, current_transitions=None):
    if current_transitions is None:
        current_transitions = []
    for state in current_state:
        if state in simulation["transitions"]:
            for transition in simulation["transitions"][state]:
                current_transitions.append(transition|{'source': state})
    return current_transitions

### UTILITIES ###

def count_clamps(current_state):
    # Loop through clamps to find how many there are and how many are free.
    i_clamp = n_free = 0
    while True:
        # Collect clamp state names with the clamp index i_clamp
        clamp_sensor_states = [state_name for state_name in current_state if state_name[-2] == str(i_clamp)]
        if len(clamp_sensor_states) == 0:
            # Covered all clamps
            break
        if all(state[0] == 'I' for state in clamp_sensor_states):
            # Clamp is free if all sensor states begin with 'I'.
            n_free += 1
        i_clamp += 1
    n_clamps = i_clamp - 1
    
    return n_clamps, n_free
