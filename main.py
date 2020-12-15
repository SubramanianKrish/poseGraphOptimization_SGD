import utils
import optim
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load MIT 3500 data
    data_path_vertex    = "data/M3500_P_toro_vertex.graph"
    data_path_edge      = "data/M3500_P_toro_edges.graph"
    
    # data_path_vertex    = "data/INTEL_P_toro_vertex.graph"
    # data_path_edge      = "data/INTEL_P_toro_edges.graph"
    temp, meas, inf_mat = utils.load_dataset(data_path_vertex, data_path_edge)
    newCost = utils.alternateCalcCost(temp, meas, inf_mat)
    print(np.sum(newCost))
    pose_init = temp.T

    print(pose_init.shape, meas.shape, inf_mat.shape)

    # Generate covariance from information matrices
    inf_mat_reshaped = np.transpose(inf_mat, axes=[2,0,1])
    covariance = np.linalg.inv(inf_mat_reshaped)

    
    optim.SGDOptimizeGraph(pose_init, meas, covariance, inf_mat)
    cost = utils.calculate_costs(X_init, meas, inf_mat)

    """
    optim.sgd(X_init.T.flatten(), meas, inf_mat)
    """

if __name__ == "__main__":
    np.set_printoptions(suppress= True)
    main()
