import argparse
import d3rlpy
import torch
import yaml
import json
import os
import numpy as np
import pandas as pd
import random
import mzutils
from zipfile import ZipFile
from gym import spaces, Env
import wandb
import pytorch_lightning as pl
from datetime import datetime
import sys
sys.path.insert(0,'..')
from utils import *



class MLP(pl.LightningModule):
    def __init__(self, normalize, max_observations, min_observations, max_actions, min_actions, clamping_max_actions, clamping_min_actions, input_size, output_size, hidden_size, num_layers, bias, weight_decay, clamping, io_type):
        super().__init__()
        self.normalize = normalize
        self.max_observations = max_observations
        self.min_observations = min_observations
        self.max_actions = max_actions
        self.min_actions = min_actions
        self.register_buffer('clamping_max_actions', torch.from_numpy(clamping_max_actions))
        self.register_buffer('clamping_min_actions', torch.from_numpy(clamping_min_actions))
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.weight_decay = weight_decay
        self.clamping = clamping
        self.io_type = io_type
        hidden2output = []
        hidden2output.append(torch.nn.Linear(input_size, hidden_size, bias=bias))
        hidden2output.append(torch.nn.RReLU())
        for i in range(self.num_layers-1):
            hidden2output.append(torch.nn.Linear(hidden_size, hidden_size, bias=bias))
            hidden2output.append(torch.nn.RReLU())
        hidden2output.append(torch.nn.Linear(hidden_size, output_size, bias=bias))
        self.mlp = torch.nn.Sequential(*hidden2output)
        self.save_hyperparameters()

    def build_input(self, observations, actions, rewards):
        """
        observations # (bs, L, observations_shape)
        actions # (bs, L, actions_shape)
        rewards # (bs, L)
        concat together to return (r, a, s)
        """
        if self.io_type == 1:
            return observations
        elif self.io_type == 2:
            return torch.cat((torch.unsqueeze(rewards, 2), actions, observations), dim=2) 

    def forward(self, input):
        """
        input should be of a torch tensor of size [bs, L, input_shape] with the same device as the model.
        """
        output = self.mlp(input) # (bs, L, output_size)
        if self.clamping:
            output = torch.clamp(output, min=self.clamping_min_actions, max=self.clamping_max_actions)
        return output

    def forward_batch_loss(self, batch):
        observations = batch['observations'] # (bs, L, observation_shape)
        actions = batch['actions'] # (bs, L, action_shape)
        rewards = batch['rewards'] # (bs, L)
        input = self.build_input(observations, actions, rewards)
        output = self.forward(input) # (bs, L, output_size)
        loss = torch.nn.functional.mse_loss(output, actions)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_batch_loss(batch)
        self.log(f'training_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.forward_batch_loss(batch)
        self.log(f'val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay)
        return optimizer


if __name__ == "__main__":
    # ----------- load configs ------------
    # overwrite args with configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = 1 if torch.cuda.is_available() else 0
    import yaml
    with open('../current.yaml', 'r') as fp:
        config_dict = yaml.safe_load(fp)
    model_name = config_dict['mlp_model_name']
    num_layers = config_dict['mlp_num_layers']
    hidden_size = config_dict['mlp_hidden_size']
    bias = config_dict['mlp_bias']
    weight_decay = config_dict['weight_decay']
    clamping = config_dict['mlp_clamping']


    normalize = config_dict['mlp_normalize']
    include_t = config_dict['include_t']
    uss_subtracted = config_dict['uss_subtracted']
    reward_on_ess_subtracted = config_dict['reward_on_ess_subtracted']
    reward_on_steady = config_dict['reward_on_steady']
    reward_on_absolute_efactor = config_dict['reward_on_absolute_efactor']
    reward_on_actions_penalty = config_dict['reward_on_actions_penalty']

    n_epochs = config_dict['mlp_n_epochs']
    log_dir = config_dict['log_dir']
    dataset_folder = config_dict['dataset_folder']
    num_of_seeds = config_dict['num_of_seeds']
    save_top_k = config_dict['save_top_k']
    io_type = config_dict['io_type']
    mzutils.mkdir_p(log_dir)
    # ----------- load configs ------------

    lbd_data_obj = DataObj(dataset_folder=dataset_folder, eval_size=0.1, shuffle=True, normalize=normalize, include_t=include_t, uss_subtracted=uss_subtracted, reward_on_ess_subtracted=reward_on_ess_subtracted, reward_on_steady=reward_on_steady, reward_on_absolute_efactor=reward_on_absolute_efactor, reward_on_actions_penalty=reward_on_actions_penalty)
    dataset_d4rl, dataset_d4rl_eval = lbd_data_obj.get_dataset()
    max_observations, min_observations, max_actions, min_actions, clamping_max_actions, clamping_min_actions = lbd_data_obj.max_observations, lbd_data_obj.min_observations, lbd_data_obj.max_actions, lbd_data_obj.min_actions, lbd_data_obj.clamping_max_actions, lbd_data_obj.clamping_min_actions # for normalization, denormalization and clamping
    training_mdptorchdataset = MDPTorchDataset(dataset_d4rl)
    eval_mdptorchdataset = MDPTorchDataset(dataset_d4rl_eval)
    observation_shape = max_observations.shape[0]
    action_shape = max_actions.shape[0]
    output_size = action_shape
    if io_type == 1:
        input_size = observation_shape
    elif io_type == 2:
        input_size = 1 + output_size + observation_shape # (r, a, s)
    else:
        raise ValueError("Unsupported io_type: {}".format(io_type))
    
    rnn_ts_model = MLP(normalize=normalize, max_observations=max_observations, min_observations=min_observations, max_actions=max_actions, min_actions=min_actions, clamping_max_actions=clamping_max_actions, clamping_min_actions=clamping_min_actions, 
        input_size=input_size, output_size=output_size, hidden_size=hidden_size, 
        num_layers=num_layers, bias=bias, weight_decay=weight_decay, clamping=clamping, io_type=io_type)

    logger_name = f'hidden_size={hidden_size}-hidden_size={hidden_size}'
    checkpointing = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='val_loss', save_top_k=save_top_k, mode='min', 
        auto_insert_metric_name=True, filename=logger_name+'{epoch}-{step}-{train_loss:.4f}-{val_loss:.4f}')
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    project_title = model_name + '_atropine_'# + current_time
    trainer = pl.Trainer(max_epochs=n_epochs, logger=pl.loggers.WandbLogger(project=project_title, name=logger_name, save_dir=log_dir),
        auto_scale_batch_size=True,default_root_dir=log_dir, callbacks=[checkpointing], check_val_every_n_epoch=10, gpus=gpus)
    trainer.fit(rnn_ts_model, torch.utils.data.DataLoader(training_mdptorchdataset, batch_size=8, num_workers=4), torch.utils.data.DataLoader(eval_mdptorchdataset))
