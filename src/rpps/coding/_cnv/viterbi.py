from ..coding import Convolutional
from .. import Stream
from .. import bitarray

import numpy as np

class Viterbi(Convolutional):
    def encode(self, data: Stream):
        """
        SOVA (Soft-Output Viterbi Algorithm) encoding
        :param data: stream of arbitrary data
        :param rate: encoding rate (e.g. 7/8)
        :return: encoded data and trellis diagram
        """
        # Calculate the number of systematic bits and redundant bits
        N_systematic = int(self._rate * len(data.bitarray))
        N_redundant = len(data.bitarray) - N_systematic

        # Initialize the trellis diagram
        T = np.zeros((2, 2), dtype=int)
        T[0, :] = [1, 0]  # initial state

        # Encode the data
        encoded_data = bitarray()
        for i in range(len(data.bitarray)):
            if i < N_systematic:
                x = data.bitarray[i]
            else:
                x = np.random.randint(2)  # add a random bit for redundancy
            y = T[:, -1].copy()  # get the previous state
            T[:, -1] = [x, x ^ y[0]]  # update the trellis diagram
            encoded_data.append(int(y[0]))
            encoded_data.append(int(y[1]))
            #encoded_data.extend([int(y[0]), int(y[1])])

        return encoded_data

    def decode(self, encoded_data):
        """
        SOVA (Soft-Output Viterbi Algorithm) decoding
        :param encoded_data: encoded data
        :param rate: encoding rate (e.g. 7/8)
        :return: decoded data
        """
        # Calculate the number of systematic bits and redundant bits
        N_systematic = int(self._rate * len(encoded_data.bitarray))
        N_redundant = len(encoded_data.bitarray) - N_systematic

        # Initialize the trellis diagram
        T = np.zeros((2, 2), dtype=int)
        T[0, :] = [1, 0]  # initial state

        # Define the transition probabilities (trellis diagram)
        P = np.array([[1/2, 1/2], [1/2, 1/2]])  # for a rate-1/2 convolutional code
        P = np.kron(P, np.ones((2, 2)))  # repeat the transition probabilities


        # Initialize the likelihoods and state sequences
        L = np.zeros((len(encoded_data.bitarray), 2))  # likelihoods
        S = np.zeros(len(encoded_data.bitarray), dtype=int)  # state sequence

        # Iterate over the codeword
        for i in range(len(encoded_data.bitarray)):
            # Calculate the likelihoods for each possible state
            #L[i, :] = P[T[:, -1].copy(), :] * (encoded_data.bitarray[i] == T[-1, :])
            L[i, :] = P[:, S[i]] * (encoded_data.bitarray[i] == T[0, :])

            # Update the state sequence and trellis diagram
            S[i] = np.argmax(L[i])
            T[:, -1] = T[S[i], :]  # update the trellis diagram

        # Extract the decoded data from the state sequence
        decoded_data = bitarray()
        for i in range(N_systematic):
            decoded_data.append(S[N_systematic + i])

        return decoded_data
