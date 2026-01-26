# no drift
import numpy as np
import matplotlib.pyplot as plt
import random
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=300)

def markoff(state, n_spatial=20):
    """
    One step of markoff process
    """
    for i in range(len(state)): # different spatial position
        for _ in range(int(state[i])): # different particles in the same position
            r = random.random() # random number between 0 and 1, smaller than 1/3: left, larger than 2/3: right
            if i!=0 and i!=n_spatial-1: # not boundary
                if r<1/3: # left
                    state[i] -= 1
                    state[i-1] += 1
                elif r>2/3: # right
                    state[i] -= 1
                    state[i+1] += 1
            elif i==0: # left boundary, reflective
                if r<1/3 or r>2/3:
                    state[i] -= 1
                    state[i+1] += 1
            elif i==n_spatial-1:
                if r<1/3 or r>2/3: # left
                    state[i] -= 1
                    state[i-1] += 1
    return state

def markoff_reversal(state, n_spatial=20):
    """
    One step of markoff reversal process
    """
    for i in range(len(state)): # different spatial position
        for _ in range(int(state[i])): # different particles in the same position
            r = random.random() # random number between 0 and 1, smaller than 1/3: left, larger than 2/3: right
            if i!=0 and i!=n_spatial-1: # not boundary
                if r<1/3: # left
                    state[i] -= 1
                    state[i+1] += 1
                elif r>2/3: # right
                    state[i] -= 1
                    state[i-1] += 1
            elif i==0: # left boundary, reflective
                if r<1/3 or r>2/3:
                    state[i] -= 1
                    state[i+1] += 1
            elif i==n_spatial-1:
                if r<1/3 or r>2/3: # left
                    state[i] -= 1
                    state[i-1] += 1
    return state

def forward_process(initial_state, n_step):
    """
    Forward process using markoff chain
    initial_state: (n_spatial,)
    n_step: int
    return: list of (n_spatial,)
    """
    state = initial_state
    for _ in range(n_step):
        state = markoff(state)
    return state

def backward_process(initial_state, n_step):
    """
    Backward process using markoff chain
    state: (n_spatial,)
    n_step: int
    return: list of (n_spatial,)
    """
    state = initial_state
    for _ in range(n_step):
        state = markoff_reversal(state)
    return state

def average_state(state):
    """
    Average state from a list of states
    state_list: list of (n_spatial,)
    return: (n_spatial,)
    """
    res = 0
    for i in range(len(state)):
        res += state[i]*i
    return res/sum(state)


if __name__ == "__main__":
    n_spatial = 20
    n_particle = 1000
    Forward_state = np.zeros(n_spatial)
    Forward_state[n_spatial//2] = n_particle
    Backward_state = np.ones(n_spatial)*int(n_particle/n_spatial)
    # print(Backward_state)

    print("Forward diffusion process\n")
    print("Forward initial average:",average_state(Forward_state))
    res1 = []
    res1.append(Forward_state)
    temp = Forward_state.copy()
    for _ in range(5):
        temp = forward_process(temp, 5)
        res1.append(temp)
        print("state:",temp)
        print("state average:",average_state(temp))

    print("\nBackward anti-diffusion process\n")
    print('Backward initial state:',temp)
    print("Backward initial average:",average_state(temp))
    res2 = []
    for _ in range(5):
        temp = backward_process(temp, 50)
        res2.append(temp)
        print("state:",temp)
        print("state average:",average_state(temp))

    print("\nPure backward anti-diffusion process\n")
    res3 = []
    res3.append(Backward_state)
    temp = Backward_state.copy()
    for _ in range(5):
        temp = backward_process(temp, 5)
        res3.append(temp)
        print("state:",temp)
        print("state average:",average_state(temp))
