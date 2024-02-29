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

################################ Input parameters begin ####################
nao_max = 19
graph_data_path = '/public/home/cwzhang/acd47nd21x/zhangchangwei/HamNet/graph'
read_openmx_path = '.'
max_SCF_skip = 200 # default is 200
scfout_paths = "/public/home/cwzhang/acd47nd21x/zhangchangwei/HamNet/PC/" # "/data/home/yzhong/EPC_test/MoS2_calculated/MoS2_*/"
dat_file_name = "Si.dat"
std_file_name = "log"
scfout_file_name = "Si.scfout"
################################ Input parameters end ######################

if nao_max == 14:
    basis_def = basis_def_14
elif nao_max == 19:
    basis_def = basis_def_19
else:
    raise NotImplementedError

graphs = dict()
if not os.path.exists(graph_data_path):
    os.makedirs(graph_data_path)
scfout_paths = glob.glob(scfout_paths)
scfout_paths = natsort.natsorted(scfout_paths)

for idx, scf_path in enumerate(tqdm(scfout_paths)):
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
            z = atomic_numbers = np.array([Element[s].Z for s in species])
            coordinates = np.array([float(pos) for pos in coordinates]).reshape(-1, 3)/au2ang
    except:
        continue
    
    # read hopping parameters
    os.system(os.path.join(read_openmx_path, 'read_openmx') + " " + f_sc)
    if not os.path.exists("./HS.json"):
        continue
    
    with open("./HS.json",'r') as load_f:
        load_dict = json.load(load_f)
        pos = np.array(load_dict['pos'])
        edge_index = np.array(load_dict['edge_index'])
        inv_edge_idx = np.array(load_dict['inv_edge_idx'])
        #
        Hon = load_dict['Hon'][0]
        Hoff = load_dict['Hoff'][0]
        Son = load_dict['Son']
        Soff = load_dict['Soff']
        nbr_shift = np.array(load_dict['nbr_shift'])
        cell_shift = np.array(load_dict['cell_shift'])
        
        # Find inverse edge_index
        if len(inv_edge_idx) != len(edge_index[0]):
            print('Wrong info: len(inv_edge_idx) != len(edge_index[0]) !')
            sys.exit()

        #
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        H = np.zeros((num_sub_matrix, nao_max**2))
        S = np.zeros((num_sub_matrix, nao_max**2))
        
        for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hon, Son)):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src = z[i]
            mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H[i][mask] = np.array(sub_maxtrix_H)
            S[i][mask] = np.array(sub_maxtrix_S)
        
        num = 0
        for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hoff, Soff)):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
            mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H[num + len(z)][mask] = np.array(sub_maxtrix_H)
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

graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
np.savez(graph_data_path, graph=graphs)
