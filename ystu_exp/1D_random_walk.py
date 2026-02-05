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

def different_transform_matrix():
    """
    This is a different local transformation matrix for 1D data.
    Having prob=1/4 go left or right, otherwise stay in place.
    """
    mat = np.zeros((n_spatial, n_spatial))
    for i in range(n_spatial):
        if i!=0 and i!=n_spatial-1:
            mat[i, i-1] = 1/2
            mat[i, i+1] = 1/2
        elif i==0:
            mat[i, i+1] = 1
        elif i==n_spatial-1:
            mat[i, i-1] = 1
    return mat.transpose()


initial_state = np.zeros(n_spatial)
initial_state[n_spatial // 2] = 1.0  # Start with all probability at the center

T = transform_matrix()
T_10times = np.identity(n_spatial)
for _ in range(500):
    T_10times = np.dot(T, T_10times)
print(T_10times)

res = []
res.append(initial_state)
temp_state = initial_state
for step in range(5):
    temp_state = np.dot(T_10times, temp_state)
    # print('step', step+1, temp_state)
    res.append(temp_state)

#画图
plt.figure()
for i in range(len(res)):
    plt.plot(res[i],label=f'step {i+1}')
plt.xlim(0, n_spatial)
plt.ylim(0, 0.1)
plt.xlabel('step')
plt.ylabel('probability')
plt.legend()
plt.savefig('./figure/1D_random_walk_large_step.png')