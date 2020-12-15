import numpy as np
import time

from utils import calcJacBlocks, Rot, mod2pi, alternateCalcCost, plot_states
from utils import tfMat, rotation_from_tfMat, tfVec
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

"This is too slow :cry:"
def sgd(states, meas, inf_m, n_iter=100, alpha=1e-10):
    """
    Input:
    states [3nx1] : x_0, y_0, th_0, x_1, y_1 ... x_3499, y_3499, th_3499
    meas   [mx5]  : m constraints of format [i, j, dx, dy, dth]
    inf_m  [3,3,m]: m 3x3 matrices representing inverse of covariance of measurement
    """
    print(np.sum(alternateCalcCost(states.reshape(-1,3).T, meas, inf_m)))

    for _ in range(n_iter):
        d = np.zeros(states.shape)
        
        # Compute M_inv at every state estimate
        J = np.zeros((3*meas.shape[0], states.shape[0]))
        for index, m in enumerate(meas):
            i, j = m[:2].astype(int)
            J[3*index: 3*index+3, 3*i: 3*i+3], J[3*index: 3*index+3, 3*j: 3*j+3] = calcJacBlocks(states[3*i: 3*i+3], states[3*j: 3*j+3], m)
        sig_inv = block_diag(*([inf_m[:,:,i] for i in range(inf_m.shape[2])]))
        M = np.diag(np.matmul(np.matmul(J.T, sig_inv), J))
        M_inv_vector = 1/M
        M_inv = np.diag(M_inv_vector)

        # Iterate through every measurement to calculate d
        for index, m in enumerate(meas):
            i, j    = m[:2].astype(int)
            sig_inv_i = inf_m[:,:,index]

            # Calculate jacobian
            # jac_i, jac_j = calcJacBlocks(states[3*i: 3*i+3], states[3*j: 3*j+3], m)
            J_i      = J[3*index: 3*index+3, :]#np.hstack((jac_i, jac_j))

            # Calculate residual
            R_i, t_i = Rot(states[3*i+2]), states[3*i: 3*i+2].reshape(-1,1)
            _, t_j   = Rot(states[3*j+2]), states[3*j: 3*j+2].reshape(-1,1)

            fx       = np.matmul(R_i.T, t_j - t_i)
            fx       = np.append(fx, mod2pi(states[3*j+2] - states[3*i+2])).reshape(-1,1)
            residual = m[2:].reshape(-1,1) - fx
            residual[2,0] = mod2pi(residual[2,0])

            d_ij = 2*alpha* np.matmul(M_inv, np.matmul(np.matmul(J_i.T, sig_inv_i), residual)) 
            
            # Update appropriate part of descent vector with d_ij
            # d[3*i: 3*i+3] += d_ij[0:3,0]
            # d[3*j: 3*j+3] += d_ij[3:6,0]
            
            # states[3*i: 3*i+3] += d_ij[0:3,0]
            # states[3*j: 3*j+3] += d_ij[3:6,0]

            d+=d_ij

            # if (np.linalg.norm(d_ij) > 1):
            #     print(d_ij)
            #     import ipdb; ipdb.set_trace()
        
        # Update state vector with d
        states += d

        # Ensure angles are within -pi and pi
        states = states.reshape(-1,3).T
        states[2,:] = mod2pi(states[2,:])
        # plot_states(states[0,:], states[1,:], states[2,:])
        print(np.sum(alternateCalcCost(states, meas, inf_m)))
        states = states.T.flatten()
    
    return states

def SGDOptimizeGraph(p, meas, cov, inf_m, n_iter=200):
    """
    Input:
    p   [nx3]  : n states representing a global pose x, y, theta
    meas[mx5]  : m measurements representing a relative measurement i, j, dx, dy, dth
    cov [mx3x3]: m covariance matrices each representing a measurement covariance 
    """

    # cost = []
    start_time = time.time()
    for iter in range(1, n_iter):
        if(iter == 101):
            print(time.time() - start_time)

        gamma = np.full((3,1), 1e12)    # Asssuming 1e12 is more or less infinity

        # Update M approximation
        M = np.zeros(p.shape)
        for index, m in enumerate(meas):
            a, b = m[:2].astype(int)
            sigma_ab = cov[index]

            # Generate matrix versions of the transformations
            P_a = tfMat(p[a])

            R = rotation_from_tfMat(P_a)
            W = np.linalg.inv(np.matmul(np.matmul(R, sigma_ab), R.T))
            diag_W = np.diag(W)
            M[a+1: b+1, :] += diag_W
            gamma = np.min(np.hstack((gamma, diag_W.reshape(-1,1))), axis=1).reshape(-1,1)
        
        # Perform modified SGD
        for index, m in enumerate(meas):
            a, b = m[:2].astype(int)
            sigma_ab = cov[index]
            P_a = tfMat(p[a])
            R = rotation_from_tfMat(P_a)

            t_ab = m[2:]
            T_ab = tfMat(t_ab)
            P_b_dash = np.matmul(P_a, T_ab)
            
            r = tfVec(P_b_dash) - p[b].reshape(-1,1)
            r[2] = mod2pi(r[2])  #correct the angle warp

            d = 2* np.matmul(np.linalg.inv(np.matmul(np.matmul(R.T, sigma_ab), R)), r)

            alpha = 1/(iter*gamma)
            total_weight = np.sum(1/M[a+1:b+1], axis=0)
            beta = (b-a)*d*alpha
            
            # Some clipping heuristic?
            mask = np.abs(beta) > np.abs(r)
            beta[mask] = r[mask]
            
            # Get the pose corrections for the current measurements
            d_pose_individual = (beta.flatten()/M[a+1:b+1])/total_weight
            d_pose_cumulative = np.cumsum(d_pose_individual, axis=0)
            
            # Update poses
            p[a+1:b+1] += d_pose_cumulative
            p[b+1:] += d_pose_cumulative[-1]

        # if(iter%20 == 0):
        #     print("20 Iterations ended. wow.")
        # plot_states(p[:,0], p[:,1], p[:,2], iter)
        # print(np.sum(alternateCalcCost(p.T, meas, inf_m)))
        # cost.append(np.sum(alternateCalcCost(p.T, meas, inf_m)))
    # plt.plot(cost)
    # plt.show()
    
