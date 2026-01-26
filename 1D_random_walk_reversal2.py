import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=300)


n_spatial = 100

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

def average_position(state):
    """
    Average position of a state
    state: (n_spatial,)
    return: float
    """
    res = 0
    for i in range(len(state)):
        res += state[i]*i
    return res/sum(state)


if __name__ == "__main__":

    initial_state = np.zeros(n_spatial)
    initial_state[n_spatial // 2] = 1.0  # Start with all probability at the center
    Backward_state = np.ones(n_spatial)/n_spatial

    T = transform_matrix()
    T_inv = np.linalg.inv(T)
    print("T dot T_inv:\n", np.dot(T, T_inv))

    print("initial state:", initial_state)
    print("average position:", average_position(initial_state))
    temp = initial_state
    for i in range(50):
        temp = np.dot(T, temp)
        if i%10==9:
            print('step', i+1, ", with state:", temp)
            print("average position:", average_position(temp))

    print("\nNow inverse process:\n")
    print("initial state:", temp)
    print("average position:", average_position(temp))
    for i in range(50):
        temp = np.dot(T_inv, temp)
        if i%10==9:
            print('inverse step', i+1, ", with state:", temp)
            print("average position:", average_position(temp))

    print("\nCompare with different initial state (uniform):\n")
    print("Backward initial state:", Backward_state)
    print("average position:", average_position(Backward_state))
    temp = Backward_state
    for i in range(50):
        temp = np.dot(T_inv, temp)
        if i%10==9:
            print('step', i+1, ", with state:", temp)
            print("average position:", average_position(temp))