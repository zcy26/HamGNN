'''
Descripttion: The scripts used to generate the input file graph_data.npz for HamNet.
version: 0.1
Author: Yang Zhong
Date: 2022-11-24 19:07:54
LastEditors: Yang Zhong
LastEditTime: 2023-01-16 14:52:09
'''

import json
import numpy as np
import os
import sys
from torch_geometric.data import Data
import torch
import glob
import natsort
from tqdm import tqdm
import re
from pymatgen.core.periodic_table import Element
from utils import *
from scipy.optimize import curve_fit

################################ Input parameters begin ####################
nao_max = 19  # maximum number of atomic orbitals
graph_data_path = '/public/home/cwzhang/acd47nd21x/zhangchangwei/HamNet/graph'  # path to save the graph data
read_openmx_path = '.'  # path of a binary executable "read_openmx"
max_SCF_skip = 200 # default is 200
scfout_paths = "/public/home/cwzhang/acd47nd21x/zhangchangwei/HamNet/PC/" # "/data/home/yzhong/EPC_test/MoS2_calculated/MoS2_*/"  # path of the DFT output
dat_file_name = "Si.dat"  # name of the dat file for DFT
std_file_name = "log"
scfout_file_name = "Si.scfout"  # name of the scfout file of DFT
weight_hamiltonian = True  # whether to weight to off-site hamiltonian through distance
weight_param = "fit"  # parameter used in the weight
################################ Input parameters end ######################


def calculate_weight(data_set, a="fit"):
    '''Calculate the weight applied to each matrix element.
    A weight proportional to exp(-a*rij) is added to each off-site matrices.
    The weights are normalized so that the maximum weight equals 1.'''
    # Extract rij and off-site H data
    dist_all = []
    n_edge_all = []  # n_edge_all[i] is the n_edge of struct i
    Hoff_mean_all = []
    for sample in list(data_set.values()):
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
        # assert len(weight_i) == data_set[i].edge_index.shape[1]
        data_set[i].weights = weight_i.reshape(-1, 1).contiguous()
        weighted_Hoff = (data_set[i].weights * data_set[i].Hoff)
        data_set[i].weighted_hamiltonian = torch.cat([data_set[i].Hon, weighted_Hoff]).contiguous()



if nao_max == 14:
    basis_def = basis_def_14  # defined in utils.py
elif nao_max == 19:
    basis_def = basis_def_19
else:
    raise NotImplementedError

graphs = dict()
if not os.path.exists(graph_data_path):
    os.makedirs(graph_data_path)
scfout_paths = glob.glob(scfout_paths)  # glob returns a list of pathnames matching the pattern given in scfout_paths
scfout_paths = natsort.natsorted(scfout_paths)  # natsort sorts a string list with human-friendly order (numbers sorted correctly)

for idx, scf_path in enumerate(tqdm(scfout_paths)):  # tqdm creates a progress bar on stdout
    # file paths
    f_sc = os.path.join(scf_path, scfout_file_name)
    f_std = os.path.join(scf_path, std_file_name)
    f_dat = os.path.join(scf_path, dat_file_name)
    f_H0 = os.path.join(scf_path, "overlap.scfout")
    
    # read energy
    try:
        with open(f_std, 'r') as f:
            content = f.read()
            Enpy = float(pattern_eng.findall((content).strip())[0][-1])
            max_SCF = int(pattern_md.findall((content).strip())[-1][-1])
    except:
        continue
    
    # check if the calculation is converged
    if max_SCF > max_SCF_skip:
        continue  
    
    # Read crystal parameters
    try:
        with open(f_dat,'r') as f:
            content = f.read()
            speciesAndCoordinates = pattern_coor.findall((content).strip())
            latt = pattern_latt.findall((content).strip())[0]
            latt = np.array([float(var) for var in latt]).reshape(-1, 3)/au2ang
    
            species = []
            coordinates = []
            for item in speciesAndCoordinates:
                species.append(item[0])
                coordinates += item[1:]
            z = atomic_numbers = np.array([Element[s].Z for s in species])  # Z number (species) for each atom
            coordinates = np.array([float(pos) for pos in coordinates]).reshape(-1, 3)/au2ang
    except:
        continue
    
    # read H & S
    os.system(os.path.join(read_openmx_path, 'read_openmx') + " " + f_sc)  # an executable read_openmx will run to create the json
    if not os.path.exists("./HS.json"):
        continue
    
    with open("./HS.json",'r') as load_f:
        load_dict = json.load(load_f)
        pos = np.array(load_dict['pos'])  # (n_atoms, 3)  # pos of atoms
        edge_index = np.array(load_dict['edge_index'])  # (2, n_edges), each col is source node idx &  dest node idx
        inv_edge_idx = np.array(load_dict['inv_edge_idx'])  # (n_edges), each elem is the idx of edge in the edge_idx
        #
        Hon = load_dict['Hon'][0]  # on-site Hamiltonian with n_atoms sub matrices, each of size (n_orb, n_orb) (n_orb may differ from each other)
        Hoff = load_dict['Hoff'][0]  # off-site Hamiltonian with n_edges sub matries, each of size (n_orb_src, n_orb_dest)
        Son = load_dict['Son']  # overlap matrices
        Soff = load_dict['Soff']
        nbr_shift = np.array(load_dict['nbr_shift'])  # (n_edges, 3) nbr_shift[i]= shift vector that is added to the position of the destination node of the edge with index i to obtain the position of its neighbor
        cell_shift = np.array(load_dict['cell_shift'])  # (n_edges, 3)
        
        # Find inverse edge_index
        if len(inv_edge_idx) != len(edge_index[0]):
            print('Wrong info: len(inv_edge_idx) != len(edge_index[0]) !')
            sys.exit()

        # expand the submatrices to nao_max**2
	# e.g.: CH4, C use s1p1, but H use s1. Submatrices are expanded to 13*13 as if all s1s2p1p2d1 is used
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]  # n_atoms + n_edges
        H = np.zeros((num_sub_matrix, nao_max**2))  # each submatrices expanded to nao_max**2
        S = np.zeros((num_sub_matrix, nao_max**2))
        
        for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hon, Son)):  # onsite
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src = z[i]  # z is atomic number (species)
            mask[basis_def[src][:,None], basis_def[src][None,:]] = 1  # The 1 elements correspond to used orbits
            mask = (mask > 0).reshape(-1)
            H[i][mask] = np.array(sub_maxtrix_H)  # sub_matrix_H is put into the  used orbits of H
            S[i][mask] = np.array(sub_maxtrix_S)
        
        num = 0
        for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hoff, Soff)): # offsite
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src, tar = z[edge_index[0,num]], z[edge_index[1,num]]  # z number of src and target
            mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H[num + len(z)][mask] = np.array(sub_maxtrix_H)  # off site matrices are stored after the len(z) onsite mats
            S[num + len(z)][mask] = np.array(sub_maxtrix_S)
            num = num + 1
    os.system("rm HS.json")
    
    # read H0
    os.system(os.path.join(read_openmx_path, 'read_openmx') + " " + f_H0)
    if not os.path.exists("./HS.json"):
        continue
    
    with open("./HS.json",'r') as load_f:
        load_dict = json.load(load_f)
        Hon0 = load_dict['Hon'][0]
        Hoff0 = load_dict['Hoff'][0]

        #
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        H0 = np.zeros((num_sub_matrix, nao_max**2))
        
        for i, sub_maxtrix_H in enumerate(Hon0):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src = z[i]
            mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H0[i][mask] = np.array(sub_maxtrix_H)
        
        num = 0
        for i, sub_maxtrix_H in enumerate(Hoff0):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
            mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H0[num + len(z)][mask] = np.array(sub_maxtrix_H)
            num = num + 1
    os.system("rm HS.json")
    
    # save in Data
    graphs[idx] = Data(z=torch.LongTensor(z),
                        cell = torch.Tensor(latt[None,:,:]),
                        total_energy=Enpy,
                        pos=torch.FloatTensor(pos),
                        node_counts=torch.LongTensor([len(z)]),
                        edge_index=torch.LongTensor(edge_index),
                        inv_edge_idx=torch.LongTensor(inv_edge_idx),
                        nbr_shift=torch.FloatTensor(nbr_shift),
                        cell_shift=torch.LongTensor(cell_shift),
                        hamiltonian=torch.FloatTensor(H),
                        overlap=torch.FloatTensor(S),
                        Hon = torch.FloatTensor(H[:pos.shape[0],:]),
                        Hoff = torch.FloatTensor(H[pos.shape[0]:,:]),
                        Hon0 = torch.FloatTensor(H0[:pos.shape[0],:]),
                        Hoff0 = torch.FloatTensor(H0[pos.shape[0]:,:]),
                        Son = torch.FloatTensor(S[:pos.shape[0],:]),
                        Soff = torch.FloatTensor(S[pos.shape[0]:,:]))
if weight_haimltonian:
    calculate_weight(graphs, a=weight_param)
graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
np.savez(graph_data_path, graph=graphs)
