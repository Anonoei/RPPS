import numpy as np

from ._code import _code

from ..blocker import block, unblock, hamming_dist


class viterbi(_code):
    """Viterbi decoder"""
    def __init__(self, num, den, constraint, generator):
        super().__init__(num, den)
        self.con = constraint
        self.gen = generator

        trellis_states = 2**(self.con-1)
        states = np.arange(trellis_states, dtype=np.uint8)
        states = np.unpackbits(states).astype(bool).reshape(-1, 8)[:, -(self.con-1):]

        # print(f"Got {len(states)} states")
        # initialize (total states; input 0,1; register bits)
        all_states = np.zeros((len(states),2,self.con), dtype=bool)
        for i, state in enumerate(all_states):
            for j, st in enumerate(state):
                all_states[i][j] = np.append(j, states[i])
                # input(f"{i},{j}: {st}")

        # initialize (total states; constraints; 0,1)
        codes = np.zeros((len(states),self.con-1,2), dtype=bool)
        # print(f"Initial codes: {codes}")
        for i, state in enumerate(all_states):
            for j, st in enumerate(state):
                for k, g in enumerate(self.gen):
                    codes[i,j,k] = np.bitwise_xor.reduce(st[g])
                # input(f"codes {i},{j} ({st}) = {codes[i,j]}")

        self.states = states
        self.all_codes = codes
        self.next_state = np.empty((len(states), 2), dtype=int)
        next_state = all_states[:, :, :-1]
        for i, state in enumerate(next_state):
            for j, b in enumerate(iterable=state):
                self.next_state[i, j] = np.where((self.states == b).all(axis=1))[0][0]

        # print(f"states: {self.states.shape}:\n{self.states.astype(int)}")
        # print(f"all_codes: {self.all_codes.shape}:\n{self.all_codes.astype(int)}")
        # print(f"next_state: {self.next_state.shape}:\n{self.next_state}")

    def decode(self, bits: np.ndarray):
        blocks = block(bits, self.den)
        trellis_length = len(blocks)

        paths = np.empty((len(self.states), trellis_length, len(self.states)), dtype=np.complex64)
        paths[:,:,:] = np.inf + np.inf*1j

        # print(f"Viterbi decode!")
        for i, state in enumerate(self.states):
            # 0: 00
            # 1: 01
            # 2: 10
            # 3: 11
            for j, blk in enumerate(blocks):
                if j == 0:
                    last = [i]
                else:
                    last = np.where(np.abs(paths[i,j-1])!=np.inf)[0]

                buffer = np.empty((len(last), len(self.states)), dtype=np.complex64)
                buffer[:,:] = np.nan - np.inf*1j

                # print(f"Checking {last}")
                for k, idx in enumerate(last):
                    # print(f"{i},{j},{idx}: {blk}")
                    code_0 = self.all_codes[idx][0]
                    code_1 = self.all_codes[idx][1]

                    dist_0 = hamming_dist(code_0, blk)
                    dist_1 = hamming_dist(code_1, blk)

                    nxt_0 = self.next_state[idx][0]
                    nxt_1 = self.next_state[idx][1]

                    if not j == 0:
                        dist_0 += paths[i,j-1,idx].real
                        dist_1 += paths[i,j-1,idx].real

                    if j < 2:
                        paths[i,j,nxt_0] = dist_0 + idx*1j
                        paths[i,j,nxt_1] = dist_1 + idx*1j
                    else:
                        buffer[k,nxt_0] = dist_0 + idx*1j
                        buffer[k,nxt_1] = dist_1 + idx*1j

                if j >= 2:
                    # print(buffer)
                    for k in range(len(self.states)):
                        valid_idx = np.where(np.abs(buffer[:,k])!=np.inf)[0]
                        valid_inp = buffer[valid_idx,k]
                        paths[i, j, k] = np.min(valid_inp)
                # print(paths[i,j])
                # print()
            break
        # print("Result:")
        # print(paths[0])
        survivor = np.empty((len(self.states), trellis_length), dtype=int)
        code = np.empty((len(self.states), trellis_length, self.den), dtype=bool)
        decoded = np.empty((len(self.states), trellis_length), dtype=bool)

        for i, state in enumerate(self.states):
            selected_path = np.min(paths[i,-1].real)
            selected_path = np.where(paths[i,-1].real==selected_path)[0][0]

            code[i,0] = self.states[0]
            decoded[i,0] = i

            # print(paths[i])

            # Reverse the trellis
            # print(f"Selected path {selected_path}")
            cur_idx = selected_path
            for j in range(len(paths[i])-1, -1, -1):
                # print(f"Decoding path[{j+1}], selected {cur_idx}")
                survivor[i,j] = cur_idx

                cur_idx = paths[i,j][cur_idx].imag.astype(int)

            for idx in range(len(survivor)):
                cur_state = survivor[i, idx]
                bit = np.where(self.next_state[cur_state]==survivor[i, idx+1])[0][0]
                code[i,idx+1] = self.all_codes[cur_state][bit]
                decoded[i,idx+1] = bit

            break

        # print(survivor[0].astype(int))
        # print(code[0].astype(int))
        # print(decoded[0].astype(int))
        return decoded[0]
