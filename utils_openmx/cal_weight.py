import numpy as np
import torch
from torch_geometric.data import Data
from scipy.optimize import curve_fit

# --------------------------- input ---------------------------------
data_dir = "/public/home/zhangchengyan/workspace/HamGNN/data_user/test/graph_data.npz"  # dir of the data
# Note that the data will be OVERWRITTEN!
a = "fit"  # parameters used in the weighting. The weights are proportional to exp(-a*rij).
# --------------------------- output --------------------------------


def calculate_weight(data_set, a="fit"):
    '''Calculate the weight applied to each matrix element.
    A weight proportional to exp(-a*rij) is added to each off-site matrices.
    The weights are normalized so that the maximum weight equals 1.'''
    # Extract rij and off-site H data
    dist_all = []
    n_edge_all = []  # n_edge_all[i] is the n_edge of struct i
    Hoff_mean_all = []
    for sample in data_set:
        Hoff = torch.abs(sample["Hoff"]).numpy()
        Hoff = np.ma.masked_equal(Hoff, 0.0)  # omit the unused orbitals
        Hoff_mean = torch.FloatTensor(np.mean(Hoff, axis=1))
        # Hoff_mean = torch.mean(torch.abs(sample["Hoff"]), dim=1)  # mean abs value of off site sub matrix
        Hoff_mean_all.append(Hoff_mean)
        src_node_pos = sample["pos"][sample["edge_index"][0]]
        tgt_node_pos = sample["pos"][sample["edge_index"][1]]  # (n_edge, 3)
        tgt_node_pos += sample["nbr_shift"]  # (n_edge, 3)
        rel_pos = tgt_node_pos - src_node_pos  # relative offsite vector
        dist = torch.linalg.norm(rel_pos, dim=1)  # distance between 2 atoms
        dist_all.append(dist)
        n_edge_all.append(sample["edge_index"].shape[1])

    Hoff_mean_all = torch.cat(Hoff_mean_all)
    dist_all = torch.cat(dist_all)
    # print(Hoff_mean_all.shape)

    if a == "fit":  # fit a
        def exp_fit(x, A, a):
            return A * np.exp(-a * x)

        popt, pcov = curve_fit(exp_fit, dist_all.numpy(), Hoff_mean_all.numpy())
        rmse = np.sqrt(torch.sum(torch.abs(Hoff_mean_all - exp_fit(dist_all, *popt))**2) / len(dist_all))
        a = popt[1]  # coeff in the weight
        print(f"The target Hamiltonian is used to fit the weights, with a={a}, and rmse={rmse}.")

    # weight the Hamiltonian
    weights = torch.exp(-a * dist_all)
    weights = weights / torch.max(weights)  # normalization
    assert torch.all(weights > 0)
    idx = 0
    for i in range(len(data_set)):
        weight_i = weights[idx: idx+n_edge_all[i]]
        idx += n_edge_all[i]
        # assert len(weight_i) == data_set[i].edge_index.shape[1]
        data_set[i].weights = weight_i.reshape(-1, 1).contiguous()
        weighted_Hoff = (data_set[i].weights * data_set[i].Hoff)
        data_set[i].weighted_hamiltonian = torch.cat([data_set[i].Hon, weighted_Hoff]).contiguous()


graph_data = np.load(data_dir, allow_pickle=True)
graph_data = graph_data['graph'].item()
graph_dataset = list(graph_data.values())
calculate_weight(graph_dataset, a=a)
graphs = dict(enumerate(graph_dataset))
np.savez(data_dir, graph=graphs)
print("done")
