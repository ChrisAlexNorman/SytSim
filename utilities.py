def name_active_states(state_values, state_names):
    named_states = []
    for index, value in enumerate(state_values):
        if value: named_states.append(state_names[index])
    return named_states

def sparsify_rate_matrix(rate_matrix, state_names, transitions={}):
    for source_idx, source_exit_rates in enumerate(rate_matrix):
        source_name = state_names[source_idx]
        for dest_idx, rate in enumerate(source_exit_rates):
            dest_name = state_names[dest_idx]

            # Only include non-zero transitions
            if str(rate) != '0':
                transition = {
                    "destination" : dest_name,
                    "rate" : rate
                }
                if not source_name in transitions: transitions[source_name] = []
                transitions[source_name].append(transition)
    return transitions

def eval_params_in_array(array, params):
    for i, row in enumerate(array):
        for j, elem in enumerate(row):
            if isinstance(elem, str):
                for param_name, param_info in params.items():
                    array[i][j] = array[i][j].replace(param_name, str(param_info["value"]))
    return array
