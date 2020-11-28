import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure import utils

class ActorPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 env=None,
                 hparams=None,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        if env is not None:
            self.env = env

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size,
                                           activation='relu',
                                           output_activation='identity')
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
            self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: 1e-3
            )
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size,
                                      activation='relu',
                                      output_activation='tanh')
            self.mean_net.to(ptu.device)
            self.optimizer = optim.Adam(
                self.mean_net.parameters(),
                self.learning_rate
            )
            self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: 1e-3
            )

        self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(obs)
        actions = self.forward(observation)
        if self.discrete:
            action = actions.sample()
        else:
            action = actions
        return action.cpu().detach().numpy()

    def update(self, ob_no_a, ob_no_b, ac_na_b, critic, **kwargs):
        if self.discrete:
            ac_na_b = ac_na_b[:,0].reshape((-1,1))
            ac_dist = self.forward(ob_no_a)
            ac_na_a = ac_dist.sample().to(dtype=torch.float32).reshape((-1,1))
            state = torch.cat((ob_no_a, ob_no_b, ac_na_b), dim=-1)
            diffQs = critic.q_net_target(state)
            _, target = diffQs.max(dim=1)
            loss = self.loss(ac_dist.logits, target.detach())
        else:
            ac_na_b = ac_na_b.reshape((-1, self.ac_dim))
            ac_na_a = self.forward(ob_no_a).to(dtype=torch.float32).reshape(ac_na_b.shape)
            state = torch.cat((ob_no_a, ob_no_b, ac_na_a , ac_na_b), dim = -1)
            loss = critic.q_net_target(state)
            loss = -torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_value_(self.mean_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        train_log = {
            'Actor Loss': ptu.to_numpy(loss)
        }

        return train_log

    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            state = self.logits_na(observation)
            state = torch.distributions.Categorical(logits=state)
            return state
        else:
            c = self.mean_net(observation)
            return c
