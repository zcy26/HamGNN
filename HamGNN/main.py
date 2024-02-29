"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-12 23:42:11 
 * @Last Modified by: Zhang Chengyan
 * @Last Modified time: 2024-1-27
 */
 """
import sklearn
import torch
import torch.nn as nn
import numpy as np
import os
import e3nn
from e3nn import o3
from GraphData.data_parsing import cif_parse, cif_parse_db
from GraphData.graph_data import graph_data_module
from input.config_parsing import read_config
from models.outputs import (Born, Born_node_vec, scalar, trivial_scalar, Force, 
                            Force_node_vec, crystal_tensor, piezoelectric, total_energy_and_atomic_forces, EPC_output)
import pytorch_lightning as pl
from models.Model import Model
from models.version import soft_logo
from pytorch_lightning.loggers import TensorBoardLogger
from models.HamGNN.net import (HamGNN_pre, HamGNN_pre2, HamGNN_out)
from torch.nn import functional as F
import pprint
import warnings
import sys
from models.utils import get_hparam_dict
from scipy.optimize import curve_fit


def prepare_data(config):
    train_ratio = config.dataset_params.train_ratio
    val_ratio = config.dataset_params.val_ratio
    test_ratio = config.dataset_params.test_ratio
    batch_size = config.dataset_params.batch_size
    split_file = config.dataset_params.split_file
    graph_data_path = config.dataset_params.graph_data_path
    if not os.path.exists(graph_data_path):
        os.mkdir(graph_data_path)
    graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
    if os.path.exists(graph_data_path):
        print(f"Loading graph data from {graph_data_path}!")
    else:
        print(f"building graph data to {graph_data_path}!")
        if config.dataset_params.database_type.lower() == 'db':
            cif_parse_db(config)
        else:
            cif_parse(config)

    graph_data = np.load(graph_data_path, allow_pickle=True)
    graph_data = graph_data['graph'].item()
    graph_dataset = list(graph_data.values())
    print("len", len(graph_dataset))
    
    # The initial feature length of the elements. (This will be deprecated in the future!)
    if config.dataset_params.database_type.lower() == 'csv':
        num_node_features = graph_dataset[0].node_attr.shape[1]
        config.dataset_params.num_node_features = num_node_features

    #if config.output_nets.HamGNN_out.weight_hamiltonian:  # calculate the weights to off-site matrices
    #    calculate_weight(graph_dataset, a=config.output_nets.HamGNN_out.weight_param)

    graph_dataset = graph_data_module(graph_dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, 
                                        batch_size=batch_size, split_file=split_file)
    graph_dataset.setup(stage=config.setup.stage)

    return graph_dataset

def build_model(config):
    print("Building model")
    if config.setup.GNN_Net.lower() == 'hamgnn_pre':
        Gnn_net = HamGNN_pre(config.representation_nets)
    elif config.setup.GNN_Net.lower() == 'hamgnn_pre2':
        Gnn_net = HamGNN_pre2(config.representation_nets)
    else:
        print(f"The network: {config.setup.GNN_Net} is not yet supported!")
        quit()

    # second order tensor
    if config.setup.property.lower() in ['born', 'dielectric']:
        if config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.atom_fea_len, num_edge_features=Gnn_net.nbr_fea_len, 
                                num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=False, num_node_features=Gnn_net.num_node_pooling_features, num_edge_features=Gnn_net.num_edge_pooling_features, num_triplet_features=Gnn_net.in_features_three_body,
                                           activation=nn.Softplus(), use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'painn':
            #output_module = Born_node_vec(num_node_features=Gnn_net.num_scaler_out, activation=Gnn_net.activation, use_bath_norm=Gnn_net.use_batch_norm, bias=Gnn_net.lnode_bias,n_h=3)
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.luse_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.n_edge_features, 
                                            num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=True, num_node_features=Gnn_net.atom_fea_len, num_edge_features=Gnn_net.nbr_fea_len, 
                                           num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.hidden_channels, 
                                           num_triplet_features=Gnn_net.num_triplet_features, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3, cutoff_triplet=config.representation_nets.dimenet_triplet.cutoff_triplet, l_minus_mean=config.setup.l_minus_mean)
        else:
            quit()

    #Force
    elif config.setup.property.lower() == 'force':
        if config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = Force(num_edge_features=Gnn_net.hidden_channels, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3)
        else:
            quit()

    #piezoelectric
    elif config.setup.property.lower() == 'piezoelectric':
        if config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = piezoelectric(include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.hidden_channels,
                                          num_triplet_features=Gnn_net.num_triplet_features, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3, cutoff_triplet=config.representation_nets.dimenet_triplet.cutoff_triplet)
        else:
            quit()
            
    # scalar_per_atom
    elif config.setup.property.lower() == 'scalar_per_atom':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = scalar('mean', False, num_node_features=Gnn_net.num_node_features, n_h=2)
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('mean', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('mean', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('mean', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar('mean', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
    
    # scalar_max
    elif config.setup.property.lower() == 'scalar_max':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = scalar(
                'max', False, num_node_features=Gnn_net.num_node_features, n_h=2)
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar(
                'max', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
    
    # scalar
    elif config.setup.property.lower() == 'scalar':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len, n_h=2)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar('sum', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
        
    # Hamiltonian
    elif config.setup.property.lower() == 'hamiltonian':
        output_params = config.output_nets.HamGNN_out
        output_module = HamGNN_out(irreps_in_node = Gnn_net.irreps_node_output, irreps_in_edge = Gnn_net.irreps_edge_output, nao_max= output_params.nao_max, ham_type= output_params.ham_type,
                                         ham_only= output_params.ham_only, symmetrize=output_params.symmetrize,calculate_band_energy=output_params.calculate_band_energy,num_k=output_params.num_k,k_path=output_params.k_path,
                                         band_num_control=output_params.band_num_control, irreps_in_triplet = Gnn_net.irreps_triplet_output if Gnn_net.export_triplet else None, include_triplet=output_params.include_triplet, 
                                         soc_switch=output_params.soc_switch, nonlinearity_type = output_params.nonlinearity_type, export_reciprocal_values = output_params.export_reciprocal_values, add_H0=output_params.add_H0, weight_hamiltonian=output_params.weight_hamiltonian)

    else:
        print('Evaluation of this property is not supported!')
        quit()
    
    # Initialize post_utility
    if config.post_processing.post_utility is None:
        post_utility = None
    elif config.post_processing.post_utility.lower() == 'epc':
        post_param = config.post_processing.EPC
        post_utility = EPC_output(representation=Gnn_net, output=output_module, band_win_min=post_param.band_win_min, band_win_max=post_param.band_win_max) 
    else:
        print(f"The post processing utility: {config.post_processing.post_utility} is not yet supported!")
        quit()
    
    return Gnn_net, output_module, post_utility

def train_and_eval(config):
    data = prepare_data(config)

    graph_representation, output_module, post_utility = build_model(config)
    graph_representation.to(torch.float32)
    output_module.to(torch.float32)

    # define metrics
    losses = config.losses_metrics.losses
    metrics = config.losses_metrics.metrics
    
    # Training
    if config.setup.stage == 'fit':
        # laod network weights
        if config.setup.load_from_checkpoint and not config.setup.resume:
            model = Model.load_from_checkpoint(checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
            )   
        else:            
            model = Model(
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience,
            )
        if config.setup.deterministic:
            ckpt = pl.callbacks.ModelCheckpoint(
                filename="{epoch}",
                save_top_k=1,
                verbose=False,
                every_n_epochs=5  # save every 5 epochs
            )
        else:
            ckpt = pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.6f}",
                save_top_k=1,
                verbose=False,
                monitor='validation/total_loss',
                mode='min',
            )
        callbacks = [
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.EarlyStopping(
                monitor="training/total_loss",
                patience=config.optim_params.stop_patience, min_delta=1e-6,
            ),
           # pl.callbacks.ModelCheckpoint(
           #     filename="{epoch}-{val_loss:.6f}",
           #     save_top_k=1,
           #     verbose=False,
           #     monitor='validation/total_loss',
           #     mode='min',
           # )
            ckpt
        ]

        tb_logger = TensorBoardLogger(
            save_dir=config.profiler_params.train_dir, name="", default_hp_metric=False)    

        if config.setup.deterministic:
            trainer = pl.Trainer(
                gpus=config.setup.num_gpus,
                precision=config.setup.precision,
                callbacks=callbacks,
                progress_bar_refresh_rate=1,
                logger=tb_logger,
                gradient_clip_val = config.optim_params.gradient_clip_val,
                max_epochs=config.optim_params.max_epochs,
                default_root_dir=config.profiler_params.train_dir,
                min_epochs=config.optim_params.min_epochs,
                resume_from_checkpoint = config.setup.checkpoint_path if config.setup.resume else None
                deterministic = True  # deterministic training
            )
        else:
            trainer = pl.Trainer(
                gpus=config.setup.num_gpus,
                precision=config.setup.precision,
                callbacks=callbacks,
                progress_bar_refresh_rate=1,
                logger=tb_logger,
                gradient_clip_val = config.optim_params.gradient_clip_val,
                max_epochs=config.optim_params.max_epochs,
                default_root_dir=config.profiler_params.train_dir,
                min_epochs=config.optim_params.min_epochs,
                resume_from_checkpoint = config.setup.checkpoint_path if config.setup.resume else None
            )

        print("Start training.")
        trainer.fit(model, data)
        print("Training done.")

        # Eval
        print("Start eval.")
        results = trainer.test(model, data.test_dataloader())
        # log hyper-parameters in tensorboard.
        hparam_dict = get_hparam_dict(config)
        metric_dict = dict() 
        for result_dict in results:
            metric_dict.update(result_dict)
        trainer.logger.experiment.add_hparams(hparam_dict, metric_dict)
        print("Eval done.")
    
    # Prediction
    if config.setup.stage == 'test': 
        model = Model.load_from_checkpoint(checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
            ) 
        tb_logger = TensorBoardLogger(
            save_dir=config.profiler_params.train_dir, name="", default_hp_metric=False)

        trainer = pl.Trainer(gpus=config.setup.num_gpus, precision=config.setup.precision, logger=tb_logger)
        trainer.test(model=model, datamodule=data)

if __name__ == '__main__':
    print(soft_logo)
    configure = read_config(config_file_name='config.yaml')
    pprint.pprint(configure)
    if configure.setup.deterministic:
        pl.seed_everything(config.setup.random_seed, workers=True)
    #torch.autograd.set_detect_anomaly(True)
    #pl.utilities.seed.seed_everything(666)
    if configure.setup.ignore_warnings:
        warnings.filterwarnings('ignore')
    train_and_eval(configure)
