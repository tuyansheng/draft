import numpy as np
import matplotlib.pyplot as plt
import random
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=300)

# no drift

def transform_matrix():
    """
    This is a local transformation matrix for 1D data.
    Having prob=1/3 go left or right, otherwise stay in place.
    """
    mat = np.zeros((n_spatial, n_spatial))
    for i in range(n_spatial):
        if i!=0 and i!=n_spatial-1:
            mat[i, i-1] = 1/3
            mat[i, i] = 1/3
            mat[i, i+1] = 1/3
        elif i==0:
            mat[i, i] = 1/3
            mat[i, i+1] = 2/3
        elif i==n_spatial-1:
            mat[i, i-1] = 2/3
            mat[i, i] = 1/3
    return mat.transpose()

def markoff_step(state, n_spatial=20):
    """
    One step of markoff process
    state: (n_spatial,)
    return: (n_spatial,)
    """
    for i in range(len(state)):
        if i-1>=0 and i+1<=n_spatial-1:
            if random.random()<1/3: # go left
                state[i-1] += 1
                state[i] -= 1
            elif random.random()<1/3: # go right
                state[i+1] += 1
                state[i] -= 1
        elif i==0:
            if random.random()<1/3: # go left
                state[i] -= 1
            elif random.random()<1/3: # go right
                state[i+1] += 1
                state[i] -= 1
        elif i==n_spatial-1:
            if random.random()<1/3: # go left
                state[i-1] += 1
                state[i] -= 1
            elif random.random()<1/3: # go right
                state[i] -= 1
    return state

def markoff_step_reversal(state, n_spatial=20):
    """
    One step of markoff process
    state: (n_spatial,)
    return: (n_spatial,)
    """
    for i in range(len(state)):
        if i-1>=0 and i+1<=n_spatial-1:
            if random.random()<1/3: # left back
                state[i-1] -= 1
                state[i] += 1
            elif random.random()<1/3: # right back
                state[i+1] -= 1
                state[i] += 1
        elif i==0:
            if random.random()<1/3: # left back
                state[i] += 1
            elif random.random()<1/3: # right back
                state[i+1] -= 1
                state[i] += 1
        elif i==n_spatial-1:
            if random.random()<1/3: # left back
                state[i-1] -= 1
                state[i] += 1
            elif random.random()<1/3: # right back
                state[i] += 1
    return state

def forward_process(initial_state, n_step):
    """
    Forward process using markoff chain
    initial_state: (n_spatial,)
    n_step: int
    return: list of (n_spatial,)
    """
    state = initial_state
    res = []
    for _ in range(n_step):
        state = markoff_step(state)
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
        state = markoff_step_reversal(state)
    return state


if __name__ == "__main__":

    # Forward process using Markoff chain
    n_spatial = 20
    n_particle = 100
    Forward_state = np.zeros(n_spatial)
    Forward_state[n_spatial//2] = n_particle
    Backward_state = np.ones(n_spatial)*(n_particle/n_spatial)

    # Forward process using Markoff chain
    res1 = []
    res1.append(Forward_state)
    res2 = []
    for _ in range(5):
        state = forward_process(Forward_state, 10)
        res1.append(state)
    for _ in range(5):
        state = backward_process(Backward_state, 10)
        res2.append(state)

    #画图
    plt.figure()
    for i in range(len(res1)):
        plt.plot(res1[i],label=f'step {i+1}')
    plt.xlim(0, n_spatial)
    plt.title('Forward Process')
    plt.xlabel('position')
    plt.ylabel('number of particles')
    plt.legend()
    plt.savefig('./figure')

    plt.figure()
    for i in range(len(res2)):
        plt.plot(res2[i],label=f'step {i+1}')
    plt.xlim(0, n_spatial)
    plt.title('Backward Process')
    plt.xlabel('position')
    plt.ylabel('number of particles')
    plt.legend()
    plt.savefig('./figure')