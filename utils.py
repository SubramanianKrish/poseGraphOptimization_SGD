import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_dataset(ver_path, edge_path):
    """
    Input:  
    ver_path    : string    path to vertices file   
    edge_path   : string    path to edges/constraints file   
   
    Returns:  
    X_init      : [3xn]     np.ndarray with initial estimate of states  
    measurements: [mx5]     np.ndarray with IDin, IDout, dx, dy, dth  
    cov         : [3x3xm]   np.ndarray with inverse covariance of each measurement  
    """

    df_ver   = pd.read_csv(ver_path, delimiter=" ")
    df_edge  = pd.read_csv(edge_path, delimiter=" ")

    x_init, y_init, theta_init = df_ver['X'].to_numpy(), df_ver['Y'].to_numpy(), df_ver['theta'].to_numpy()
    dX_rel = df_edge[df_edge.columns[1:6]].to_numpy()

    meas_inf = np.zeros((3,3,dX_rel.shape[0]))
    meas_inf[0,0,:] = df_edge['I11'].to_numpy()
    meas_inf[0,1,:] = df_edge['I12'].to_numpy()
    meas_inf[0,2,:] = df_edge['I13'].to_numpy()
    meas_inf[1,0,:] = df_edge['I12'].to_numpy()
    meas_inf[1,1,:] = df_edge['I22'].to_numpy()
    meas_inf[1,2,:] = df_edge['I23'].to_numpy()
    meas_inf[2,0,:] = df_edge['I13'].to_numpy()
    meas_inf[2,1,:] = df_edge['I23'].to_numpy()
    meas_inf[2,2,:] = df_edge['I33'].to_numpy()

    # plot_states(x_init, y_init, theta_init, animate=False)
    
    return np.vstack((x_init.reshape(1,-1), y_init.reshape(1,-1), theta_init.reshape(1,-1))), dX_rel, meas_inf

def calcJacBlocks(state_i, state_j, measurement):
    """
    Calculates two jacobian blocks based on input state and measurement
    Input:
    state_i     : (3,) x_i, y_i, theta_i
    state_j     : (3,) x_j, y_j, theta_j
    measurement : (5,) i,j, x_ij, y_ij, theta_ij 
    
    Output:
    jac_i       : (3,3) differentiation of f(x) w.r.t X_i=>[x_i, y_i, theta_i]
    jac_j       : (3,3) differentiation of f(x) w.r.t X_j=>[x_j, y_j, theta_j]
    """

    jac_i = np.zeros((3,3))
    jac_i[0:2, 0:2] = np.matmul(Rot(state_i[2]), np.array([[state_j[0]-1, 0],[0, state_j[1]-1]]))
    jac_i[0:2,2]    = np.matmul(np.array([[-np.sin(state_i[2]), np.cos(state_i[2])], \
                                          [-np.cos(state_i[2]), -np.sin(state_i[2])]]), \
                                         state_j[:2].reshape(-1,1) - state_i[:2].reshape(-1,1)).flatten()
    jac_i[2,2]      = state_j[2] - 1  # <check again if buggy!>

    jac_j           = np.zeros((3,3))
    jac_j[0:2, 0:2] = np.matmul(Rot(state_i[2]), np.array([[1-state_i[0], 0],[0, 1-state_i[1]]]))
    jac_j[2,2]      = 1 - state_i[2]  # <check again if buggy!>

    return jac_i, jac_j

def calculate_costs(states, measurements, cov):
    cost = []
    for index, m in enumerate(measurements):

        i, j     = m[0:2].astype(int)
        R_i, t_i = Rot(states[2,i]), states[:2, i].reshape(-1,1)
        R_j, t_j = Rot(states[2,j]), states[:2, j].reshape(-1,1)
        t_ij     = m[2:4].reshape(-1,1)
        R_ij     = Rot(m[4])
        omega_ij = cov[:,:,index]

        rot_cost      = (log_map(np.matmul(R_ij.T, np.matmul(R_i.T, R_j))**2)*omega_ij[2,2])
        trans_cost    = maha_dist(t_ij - np.matmul(R_i.T, t_j - t_i), omega_ij[:2,:2])
        current_cost  = rot_cost + trans_cost

        cost.append(current_cost)

    return np.array(cost).squeeze()

def alternateCalcCost(states, measurements, cov):
    cost = []
    for index, m in enumerate(measurements):
        i, j     = m[0:2].astype(int)
        R_i, t_i = Rot(states[2,i]), states[:2, i].reshape(-1,1)
        _, t_j = Rot(states[2,j]), states[:2, j].reshape(-1,1)
        omega_ij = cov[:,:,index]

        fx       = np.matmul(R_i.T, t_j - t_i)
        fx       = np.append(fx, mod2pi(states[2,j] - states[2,i])).reshape(-1,1)
        residual = m[2:].reshape(-1,1) - fx
        residual[2,0] = mod2pi(residual[2,0])
        
        current_cost = np.matmul(residual.T, np.matmul(omega_ij, residual))
        cost.append(current_cost)

    return np.array(cost).squeeze()

def mod2pi(x):
    # ref: https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    return (x + np.pi) % (2 * np.pi) - np.pi

def log_map(R):
    return np.arctan2(R[1,0], R[0,0])

def maha_dist(x, I):
    return np.matmul(np.matmul(x.T, I), x)

def Rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def plot_states(x, y, theta, iter_id=0, animate=False):
    if not animate:
        plt.cla()
        plt.plot(x, y)
        plt.savefig("output/iter_"+str(iter_id)+".png", bbox_inches='tight')
        # plt.show()

    else:
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = plt.plot([], [])

        def init():
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y), np.max(y))
            return ln,

        def update(i):
            xdata.append(x[i])
            ydata.append(y[i])
            ln.set_data(xdata, ydata)
            return ln,

        ani = FuncAnimation(fig, update, frames=x.shape[0], init_func=init, blit=True, interval=20)
        plt.show()


# Olson optimization utils
def tfMat(tf_vector):
    x, y, theta = tf_vector
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), x],[np.sin(theta), np.cos(theta), y], [0, 0, 1]])
    return transformation_matrix

def rotation_from_tfMat(tf_matrix):
    R = np.copy(tf_matrix)
    R[0:2,2] = [0,0]
    return R

def tfVec(tf_matrix):
    return np.array([tf_matrix[0,2], tf_matrix[1,2], np.arctan2(tf_matrix[1,0], tf_matrix[0,0])]).reshape(-1,1)
