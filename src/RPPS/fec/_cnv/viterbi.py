from ..convolutional import Convolutional

import numpy as np

class Viterbi(Convolutional):
    delta = np.zeros((num_states, num_observations))  # Viterbi path metrics matrix initialization
    pointer = np.zeros(delta.shape, dtype=int)        # Backtrack pointers for reconstructing the best path later

    for t in range(1, num_observations):
       delta[:,t] = (delta[:,t-1].T * transition_matrix).T + emission[obs[t], :]*np.log2(fcr)  # Calculate new metrics based on transitions and emissions for code rate ratios like 7/8 or 3/4
       pointer[:, t] = np.argmax(delta[:, t])          # Record the state with highest metric value as backpointers

    end_state = np.argmax(delta[:, num_observations - 1])     # Find the state with maximum probability at last time step (decoded bits)
    viterbi_path = [end_state]                                # Start path reconstruction from this state
    for t in range(num_observations-1, 0, -1):               # Trace back to find previous states through recorded pointers
        end_state = pointer[end_state,t+1]
        viterbi_path.insert(0, int(end_state))                # Insert found state at the beginning of path list for correct order reconstruction
