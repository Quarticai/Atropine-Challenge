import numpy as np
import torch
import os
import pickle
import pytorch_lightning as pl
import yaml


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
        hidden2output = [torch.nn.Linear(input_size, hidden_size, bias=bias)]
        hidden2output.append(torch.nn.RReLU())
        for _ in range(self.num_layers-1):
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
        return torch.nn.functional.mse_loss(output, actions)

    def training_step(self, batch, batch_idx):
        loss = self.forward_batch_loss(batch)
        self.log('training_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.forward_batch_loss(batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay)


class controller:
    def __init__(self):
        self.current_algo = MLP.load_from_checkpoint('control_model/best.pt').eval()
        self.normalize = self.current_algo.normalize

    def get_input(self, state):
        """
        state is a numpy array of shape (39,) that contains
        ["step", "USS1", "USS2", "USS3", "USS4", "ESS", "E", "KF_X1", "KF_X2", 
        "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", 
        "Z12", "Z13", "Z14", "Z15", "Z16", "Z17", "Z18", "Z19", "Z20", "Z21", 
        "Z22", "Z23", "Z24", "Z25", "Z26", "Z27", "Z28", "Z29", "Z30"]
        step ranges from 0 to 59.
        the expected output is a numpy array of shape (4,),
        representing actions
        """
        with torch.no_grad():
            state = torch.from_numpy(state.reshape(1,1,state.shape[0])).to(torch.float32)
            return self.current_algo.forward(state)[0][0].numpy()
