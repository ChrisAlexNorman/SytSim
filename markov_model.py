import numpy as np
import pandas as pd
import numexpr as ne
import random
import copy
import json
import jsbeautifier
from scipy.integrate import odeint, cumtrapz
from scipy.optimize import root_scalar
import multiprocessing as mp 
import time
from clamping import apply_transition, get_feasible_transitions
from utilities import name_active_states, sparsify_rate_matrix, eval_params_in_array

### MARKOV MODEL CLASS ###

class MarkovModel:

    def __init__(self, **kwargs):
        model_spec = self._parse_model(kwargs)
        for key, value in model_spec.items():
            setattr(self, key, value)

    def _parse_model(self, model_spec):

        # Allow for empty class initialisation
        if len(model_spec) == 0:
            return model_spec

        # Input requirements
        if "transitions" not in model_spec and "rate_matrix" not in model_spec:
            raise KeyError("'transitions' or 'rate_matrix' must be specified")
        
        # Validate model for ODE simulation and prepare for stochastic
        if "rate_matrix" in model_spec:
            # Verify initial condition
            if "initial_condition" not in model_spec:
                raise KeyError(f"initial_condition not specified.")
            # Verify rate_matrix is square
            rates_shape = np.shape(model_spec["rate_matrix"])
            if len(rates_shape) != 2 or rates_shape[0] != rates_shape[1]:
                raise ValueError(f"rates is not square. Shape is {rates_shape}.")
            # Verify rate_matrix and initial_condition dimensions are equal
            rate_dim = len(model_spec["rate_matrix"])
            init_dim = len(model_spec["initial_condition"])
            if rate_dim != init_dim:
                raise ValueError(f"Rate matrix is {rate_dim}x{rate_dim} but initial condition has length {init_dim}.")
            
            # Evaluate rate_matrix with given parameters
            if "parameters" in model_spec:
                model_spec["rate_matrix"] = eval_params_in_array(model_spec["rate_matrix"], model_spec["parameters"])
            # Convert to sparse transitions if not already specified
            model_spec.setdefault(
                "transitions",
                sparsify_rate_matrix(
                    model_spec["rate_matrix"], model_spec["state_names"]
                )
            )
            # Convert initial_condition if not already specified
            model_spec.setdefault("state_names", ["S" + str(n) for n in range(0, len(model_spec["rate_matrix"]))])
            model_spec.setdefault("initial_condition_names", name_active_states(model_spec["initial_condition"], model_spec["state_names"]))

        # todo: Validate model for stochastic simulation

        # Assumed properties
        model_spec.setdefault("name", "Unnamed")
        model_spec.setdefault("simulations", [])
        model_spec.setdefault("multiprocessing", True)
        model_spec.setdefault("state_update_rule", apply_transition)
        model_spec.setdefault("rate_update_rule", get_feasible_transitions)

        return model_spec

    def _parse_simulation(self, sim_spec):
        
        # Required specifications
        if "stimuli" not in sim_spec and "timestamp" not in sim_spec:
            raise ValueError("Either timestamped 'stimuli' or 'timestamp' must be specified.")
        sim_spec.setdefault("mode", "stochastic")
        recognised_modes = ["deterministic", "stochastic"]
        if sim_spec["mode"].lower() not in recognised_modes:
            raise ValueError(f"mode '{sim_spec['mode']}' not recognised. Must be one of {recognised_modes}.")
        sim_spec["mode"] = sim_spec["mode"].lower()
        
        # Parse stimuli
        sim_spec.setdefault("stimuli", {})
        sim_spec["input_stimuli"] = sim_spec["stimuli"]
        sim_spec.setdefault("timestamp", [])
        sim_spec["stimuli"], sim_spec["timestamp"] = align_and_unpack_stimuli_values(sim_spec["input_stimuli"], sim_spec["timestamp"])
        
        if sim_spec["mode"] == "deterministic":
            sim_spec["initial_condition"] = self.initial_condition
            sim_spec["rate_matrix"] = self.rate_matrix
        elif sim_spec["mode"] == "stochastic":
            sim_spec["initial_condition"] = self.initial_condition_names
            sim_spec["transitions"] = evaluate_transitions_as_timeseries(self.transitions, sim_spec["timestamp"], sim_spec["stimuli"])
        
        # Set defaults
        sim_spec["state_names"] = self.state_names
        sim_spec.setdefault("state_update_rule", self.state_update_rule)
        sim_spec.setdefault("rate_update_rule", self.rate_update_rule)
        sim_spec.setdefault("record", self.state_names)
        sim_spec.setdefault("track_clamps", False)
        sim_spec.setdefault("runtime", time.time())
        sim_spec.setdefault("n_processes", 1)

        # Set number of simulations or required number of events
        if "n_simulations" in sim_spec:
            sim_spec["batch_size"], sim_spec["n_simulations"] = sim_spec["n_simulations"], 0
            sim_spec["n_events_required"] = {key: 0 for key in sim_spec["record"]}
        elif "n_events_required" in sim_spec:
            sim_spec["n_simulations"] = 0
            sim_spec.setdefault("batch_size", 10000)
            for required_key in sim_spec["n_events_required"].keys():
                if required_key not in sim_spec["record"]:
                    sim_spec["record"].append(required_key)
        elif sim_spec["mode"] == "stochastic":
            raise ValueError("One of 'n_simulations' or 'n_events_required' must be specified for 'stochastic' simulation")
        return sim_spec

    def import_model(self, filename):
        data = self._parse_model(read_json_model(filename))
        for key, value in data.items():
            setattr(self, key, value)
        return self

    def export_model(self, filename):
        model_spec = {key: value for key, value in vars(self).items()}
        write_json_model(filename, model_spec)
        return self

    def clear_simulations(self):
        simulations = self.simulations
        self.simulations = []
        return simulations   

    def simulate(self, **kwargs):
        simulation = self._parse_simulation(kwargs)

        # Deterministic (ODE) solution
        if simulation["mode"].lower() == "deterministic":
            # Numerically solve ODE
            states = odeint(
                ode_system,
                simulation["initial_condition"],
                simulation["timestamp"],
                args=(
                    simulation["rate_matrix"],
                    simulation["timestamp"],
                    simulation["stimuli"],
                ),
            )
            # Return results requested in 'record'   
            states = np.transpose(states)
            probability = {
                self.state_names[idx]: state
                for idx, state in enumerate(states)
                if self.state_names[idx] in simulation["record"]
            }
            simulation["probability"] = probability
            # Convert probabilities to rates:
            simulation["rate"] = {}
            for state, probability in simulation["probability"].items():
                simulation["rate"][state] = prob_to_rate(probability, simulation["timestamp"])

        # Stochastic (Gillespie) solution
        elif simulation["mode"].lower() == "stochastic":
            simulation = run_stochastic_simulations(simulation)

        simulation['runtime'] = time.time() - simulation['runtime']
        self.simulations.append(simulation)
        return simulation

### MODEL I/O ###

def read_json_model(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def write_json_model(filename, model_spec=None):
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    model_beaut = jsbeautifier.beautify(json.dumps(model_spec), opts)
    with open(filename, 'w') as file:
        file.write(model_beaut)
    return model_beaut

### SIMULATION ###

def ode_system(states, t, rate_matrix, timestamp, stimuli_values):
    '''Return rate of change of state probabilities, dp/dt, at time t given stimuli values.'''
    stimuli_at_t = {name: np.interp(t, timestamp, stimulus) for  name, stimulus in stimuli_values.items()}
    q_matrix_at_t = np.array(
        [
            [
                ne.evaluate(str(element), stimuli_at_t).item()
                for element in row
            ]
            for row in rate_matrix
        ]
    )
    q_matrix_at_t[np.diag_indices(len(rate_matrix))] = -np.sum(q_matrix_at_t, axis=1)
    return np.dot(states, q_matrix_at_t)

def run_stochastic_simulations(simulation):
    '''Calls stochastic simulator the specified number of times and processes results.'''

    # Initialise results container
    simulation["event_times"] = {key: np.array([]) for key in simulation["record"]}
    if simulation["track_clamps"]:
        simulation["free_clamps"] = np.zeros(len(simulation["timestamp"]))
        free_clamps_fused = {"timestamp": [], "n_free_clamps": []}

    with mp.Pool(simulation["n_processes"]) as pool:
        # Batch simulations until required number of events achieved
        while not event_requirements_met(simulation):
            simulation["n_simulations"] += simulation["batch_size"]
            async_results = []
            for _ in range(simulation["batch_size"]):
                async_results.append(
                    pool.apply_async(
                        stochastically_simulate,
                        args=(simulation,)
                    )
                )
            # Collect event times
            for output in async_results:
                event_times = output.get()
                for key, values in event_times.items():
                    simulation["event_times"][key] = np.append(
                        simulation["event_times"][key], (values)
                    )
                # Convert clamping events into timeseries (clamp models only)
                if simulation["track_clamps"]:
                    timestamp, n_free_clamps = calc_free_clamp_timeseries(event_times)
                    clamp_t_idx = 0
                    for out_idx, out_t in enumerate(simulation["timestamp"]):
                        while clamp_t_idx < len(timestamp)-1 and out_t > timestamp[clamp_t_idx+1]:
                            clamp_t_idx += 1
                        simulation["free_clamps"][out_idx] += n_free_clamps[clamp_t_idx]
                    # Calculate free clamps at instance of fusion
                    free_clamps_fused["timestamp"].extend(event_times["Fused"])
                    free_clamps_fused["n_free_clamps"].extend(np.interp(event_times["Fused"], timestamp, n_free_clamps))
        
    if simulation["track_clamps"]:
        # Calculate free clamps on unfused SVs
        # TODO: This approach only valid for no vesicle replenishment
        unfused_normalisation = [simulation["n_simulations"]] * len(simulation["timestamp"])
        simulation["free_clamps_unfused"] = copy.deepcopy(simulation["free_clamps"])
        # print(simulation["free_clamps"])
        for fusion_time, n_free in zip(free_clamps_fused["timestamp"], free_clamps_fused["n_free_clamps"]):
            # print(fusion_time, n_free)
            for idx, t in enumerate(simulation["timestamp"]):
                if t > fusion_time:
                    unfused_normalisation[idx] -= 1
                    simulation["free_clamps_unfused"][idx] -= n_free
        simulation["free_clamps_unfused"] /= unfused_normalisation
        
        # Normalise all free clamps over time
        simulation["free_clamps"] /= simulation["n_simulations"]

        # Process free clamps at instance of fusion
        # Calculate the midpoints between each pair of consecutive timestamps
        midpoints = (simulation["timestamp"][:-1] + simulation["timestamp"][1:]) / 2
        # Create bins with lower edge at midpoint to previous timestamp and upper edge at midpoint to next timestamp
        bins = np.concatenate([[simulation["timestamp"][0] - (midpoints[0] - simulation["timestamp"][0])], 
                            midpoints, 
                            [simulation["timestamp"][-1] + (simulation["timestamp"][-1] - midpoints[-1])]])
        # Convert to dataframe
        df = pd.DataFrame(free_clamps_fused)
        df['bin'] = np.digitize(df['timestamp'], bins) - 1
        # Calculate the mean and standard deviation for each bin
        grouped = df.groupby('bin')['n_free_clamps']
        mean_std = pd.DataFrame({
            'mean': grouped.mean(),
            'std': grouped.std()
        })
        # To match with simulation["timestamp"], reindex the DataFrame
        mean_std = mean_std.reindex(np.arange(len(simulation["timestamp"])))
        # If there are NaN values (which means no data points in those bins), fill them with zero
        mean_std = mean_std.fillna(0)
        # Create output
        simulation["free_clamps_fused"] = {
            'mean': mean_std['mean'].tolist(),
            'std': mean_std['std'].tolist()
        }

    # Convert event times to probabilities
    simulation["probability"], simulation["raw_probability"] = events_to_probs(simulation["event_times"], simulation["n_simulations"], simulation["timestamp"])

    # Convert event times to event rates
    simulation["rate"] = {}
    for state, event_times in simulation["event_times"].items():
        simulation["rate"][state] = calc_event_rate(event_times, simulation["n_simulations"])
    
    return simulation

def stochastically_simulate(simulation):
    '''Runs direct Gillespie algorithm for specified simulation including spontaneous state and rate update rules.'''
    random.seed(int(time.time() * 1e9) * mp.current_process().pid)

    # Unpack values from 'simulation'
    initial_condition = simulation["initial_condition"]
    end_time = simulation["timestamp"][-1]
    record = simulation["record"]
    state_update_rule = simulation["state_update_rule"]
    rate_update_rule = simulation["rate_update_rule"]

    # Initialise results container
    event_times = {key: [] for key in record}

    # Initialise system
    current_time = 0
    current_state = changed_state = copy.deepcopy(initial_condition)
    current_transitions = rate_update_rule(current_state, simulation)

    # Run direct Gillespie algorithm
    while current_time < end_time:

        # Update state record
        for state in changed_state:
            if state in record:
                event_times[state].append(current_time)

        # Find time waited in current state
        if len(current_transitions) == 0: break
        shifted_rate_integrals = shift_rate_integrals(current_time, current_transitions)
        rand_log = np.log(random.random())
        try:
            wait_time = root_scalar(wait_time_root_func, x0=0, bracket=[0, end_time-current_time], args=(shifted_rate_integrals, rand_log), method='brentq').root # or 'toms748' 
        except:
            # No solution for wait_time before end_time
            break
        # Advance current time to next transition time
        current_time += wait_time
        
        # find which transition occurs
        current_rates = [
            np.interp(current_time, transition["timestamp"], transition["rate"])
            for transition in current_transitions
        ]
        chosen_transition = current_transitions[
            np.searchsorted(
                np.cumsum(current_rates) / np.sum(current_rates), random.random()
            )
        ]

        # Update current state
        current_state, changed_state = state_update_rule(current_state, simulation, chosen_transition)

        # Update current transitions
        current_transitions = rate_update_rule(current_state, simulation, chosen_transition)
        
    return event_times

def event_requirements_met(simulation):
    for key, n_required in simulation["n_events_required"].items():
        if len(simulation["event_times"][key]) <= n_required:
            return False
    return True

def shift_rate_integrals(current_time, current_transitions):
    return [{
    'timestamp' : np.concatenate((
        [0], transition['timestamp'][transition['timestamp']>current_time]-current_time
        )),
    'value': np.concatenate((
        [np.interp(current_time, transition['timestamp'], transition['rate_integral'])],
        transition['rate_integral'][transition['timestamp']>current_time]
        ))
    } for transition in current_transitions]

def wait_time_root_func(wait_time, shifted_rate_integrals, rand_log):
    '''
    Return sum of rate integrals up to wait_time plus given log(rand).
    Assumes integrals shifted to current_time = 0
    '''
    func_sum = rand_log
    for integral in shifted_rate_integrals:
        func_sum += np.interp(wait_time, integral['timestamp'], integral['value']) - integral['value'][0]

    return func_sum

### PROCESSING ###

def evaluate_transitions_as_timeseries(transitions, timestamp, stimuli):
    transitions_evaluated = copy.deepcopy(transitions)
    for transition_destinations in transitions_evaluated.values():
        for transition in transition_destinations:
            # Evaluate rate expression as a string equation given parameter and stimuli values
            rate_timeseries = ne.evaluate(str(transition["rate"]), stimuli)

            # If the rate is a constant then transform it into a timeseries between the start and end times
            if rate_timeseries.ndim == 0 or np.all(rate_timeseries == 0):
                rate_timestamp = np.array([0, timestamp[-1]])
                rate_timeseries = np.repeat(rate_timeseries, 2)
            else:
                rate_timestamp = timestamp
            
            # Numerically integrate the transition rate
            rate_integral = cumtrapz(rate_timeseries, rate_timestamp, initial=0)

            # Update transition
            transition["timestamp"] = rate_timestamp
            transition["rate"] = rate_timeseries
            transition["rate_integral"] = rate_integral
    
    return transitions_evaluated

def align_and_unpack_stimuli_values(stimuli, timestamp):
    # If timestamp not specified, take timestamp from shortest stimulus.
    if len(timestamp) == 0:
        timestamp = [np.inf]
        for stimulus in stimuli.values():
            if stimulus['timestamp'][-1] < timestamp[-1]:
                timestamp = stimulus['timestamp']

    # Extract stimulus values from stimulus data, interpolated at timestamp.
    stimuli_values = {
        name: np.interp(timestamp, stimulus['timestamp'], stimulus['value'])
        for name, stimulus in stimuli.items()
    }

    return stimuli_values, timestamp

def events_to_probs(event_time_dict, n_sims, timestamp):
    raw_probability = {
        name: {
            'timestamp': np.sort(event_times),
            'probability': np.array(range(1,len(event_times)+1))/n_sims
        }
        for name, event_times in event_time_dict.items()
    }
    probability = {
        name: np.interp(timestamp, np.insert(np.sort(event_times), 0, 0), np.array(range(len(event_times)+1))/n_sims)
        for name, event_times in event_time_dict.items()
    }
    return probability, raw_probability

def calc_event_rate(event_times, n_simulations, smoothing=False):
    # todo: add smoothing and better data averaging
    hist_counts, bin_edges = np.histogram(event_times, bins='auto') # 'auto'
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    prob_values = np.cumsum(hist_counts) / n_simulations
    release_rate = np.diff(prob_values) / np.diff(bin_centres)
    return {'timestamp': bin_centres, 'rate': release_rate}

def prob_to_rate(probability, timestamp, smoothing=False):
    diff_timestamp = np.diff(timestamp)
    diff_prob = np.diff(probability)
    mid_timepoints = timestamp[:-1] + diff_timestamp / 2
    rate = diff_prob / diff_timestamp
    return {'timestamp': mid_timepoints, 'rate': rate}

def calc_free_clamp_timeseries(event_times):
    # Convert to timeseries
    timestamp = []
    n_free_clamps = []
    i_clamp = 0
    while f"{i_clamp}_free_clamps" in event_times:
        for event_time in event_times[f"{i_clamp}_free_clamps"]:
            timestamp.append(event_time)
            n_free_clamps.append(i_clamp)
        i_clamp += 1

    timestamp, n_free_clamps = zip(*sorted(zip(timestamp, n_free_clamps)))
    return timestamp, n_free_clamps
