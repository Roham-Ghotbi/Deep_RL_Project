from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import numpy as np
import itertools

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class state_marginal(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, replay_buffer, **kwargs):
        super().__init__(**kwargs)
        self.replay_buffer = replay_buffer
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        self.expect_x = torch.zeros([1])
        self.expect_x2 = torch.zeros([1])
        self.n = 0


    def forward(self, ob_no):
        # TODO: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        if isinstance(ob_no, np.ndarray):
            ob_no = ptu.from_numpy(ob_no)

        # reward = torch.tensor([- self.dist.log_prob(ob) for ob in ob_no])
        # reward = 0.5 * ((ob_no - self.expect_x).square()/self.var).sum(dim=-1) + 0.5 * torch.log(self.var.prod())
        # reward = - (-0.5*torch.matmul((ob_no-self.loc).transpose(1,0),ob_no-self.loc)/self.cov.diagonal(0) - torch.log(torch.sqrt(self.cov.diagonal(0).prod())))
        curr_var = self.expect_x2 - torch.square(self.expect_x)
        reward = ((self.expect_x2 * self.n + ob_no.square())/(self.n + 1) - ((self.expect_x * self.n + ob_no)/(self.n + 1)).square()) - curr_var
        reward = reward.mean(dim=-1)
        return reward

    # def log_prob(self,ob_no):
    #     mu_x = self.mu_net[0::2]
    #     mu_y = self.mu_net[1::2]
    #     sig_x = torch.exp(self.log_sigma_net[0::2])
    #     sig_y = torch.exp(self.log_sigma_net[1::2])
    #     self.mu = torch.cat((mu_x.unsqueeze(-1), mu_y.unsqueeze(-1)), dim=1)
    #     self.sig = torch.cat((sig_x.unsqueeze(-1), sig_y.unsqueeze(-1)), dim=1)
    #     log_prob = 0
    #     for j in range(self.n_gaus):
    #         log_prob = log_prob + torch.exp(- 0.5 * ((ob_no - self.mu[j,:]).square() / self.sig[j,:]).sum(dim=-1) - 0.5 * torch.log(self.sig[j,:].prod()))
    #     log_prob = torch.log(log_prob/self.n_gaus)
    #     return log_prob


    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # if isinstance(ob_no, np.ndarray):
        #     ob_no = ptu.from_numpy(ob_no)
        #
        # if self.replay_buffer.can_sample(256):
        #     _, _, _, states, _ = self.replay_buffer.sample(256)
        #     num_states = 256
        # else:
        #     num_states = self.replay_buffer.num_in_buffer - 2
        #     states = self.replay_buffer.next_obs[:num_states]
        #
        # self.expect_x = ptu.from_numpy(states.mean(axis=0))
        # self.expect_x2 = ptu.from_numpy(np.square(states).mean(axis=0))
        #
        # self.n = num_states


        return 0 #self.dist.entropy()
