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

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 env=None,
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
            self.cov_factor = torch.normal(-1.5, 0.05, size=(self.ac_dim,self.ac_dim), dtype=torch.float32, device=ptu.device)
            self.logstd = torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            for i in range(self.ac_dim):
                for j in range(i, self.ac_dim):
                    self.cov_factor[i, j] = self.cov_factor[j,i]
            self.logstd = nn.Parameter(self.logstd)
            self.cov_factor = nn.Parameter(self.cov_factor)
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            # self.optimizer = optim.Adam(
            #     itertools.chain([self.cov_factor],[self.logstd], self.mean_net.parameters()),
            #     self.learning_rate
            # )
            self.optimizer = optim.Adam(
                self.mean_net.parameters(),
                self.learning_rate
            )
            self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: 1e-3
            )
            # self.learning_rate_scheduler = optim.lr_scheduler.StepLR(
            #     self.optimizer,
            #     step_size=5000,
            #     gamma=0.7
            # )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
                activation='tanh',
                output_activation='identity'
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from hw1 or hw2
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(obs)
        actions = self.forward(observation)
        if self.discrete:
            # actions = c.sample()
            action = actions.sample()
        else:
            # action = c.sample()
            action = actions #.sample()
        # c = torch.distributions.Categorical(logits=state)
        # action = c.sample()
        # action = state.argmax()
        return action.cpu().detach().numpy()

    # update/train this policy
    def update(self, ob_no_a, ob_no_b, ac_na_b, critic, **kwargs):
        ac_na_b = ac_na_b.reshape((-1,self.ac_dim))
        if self.discrete:
            ac_dist = self.forward(ob_no_a)
            ac_na_a = ac_dist.sample().to(dtype=torch.float32).reshape((-1,1))
            state = torch.cat((ob_no_a, ob_no_a, ac_na_a), dim=-1)
            diffQs = critic.q_net_target(state)
            _, target = diffQs.max(dim=1)
            loss = self.loss(ac_dist.logits, target.detach())
        else:
            ac_na_a = self.forward(ob_no_a).to(dtype=torch.float32).reshape(ac_na_b.shape)
            state = torch.cat((ob_no_a, ob_no_b, ac_na_a , ac_na_b), dim = -1)
            loss = critic.q_net_target(state)
            loss = -torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_log = {
            'Actor Loss': ptu.to_numpy(loss)
        }

        return train_log

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from hw1 or hw2
        if self.discrete:
            state = self.logits_na(observation)
            state = torch.distributions.Categorical(logits=state)
            return state
        else:
            # c = torch.distributions.LowRankMultivariateNormal(self.mean_net(observation).mul(ptu.from_numpy(self.env.action_space.high)),
            #                                                   10 ** self.cov_factor,
            #                                                   10 ** self.logstd)
            c = self.mean_net(observation)
            return c
