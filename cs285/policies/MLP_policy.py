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
            action = actions.argmax()
        else:
            # action = c.sample()
            action = actions
        # c = torch.distributions.Categorical(logits=state)
        # action = c.sample()
        # action = state.argmax()
        return action.detach().numpy()

    # update/train this policy
    def update(self, ob_no_a, ob_no_b, ac_na_b, critic, **kwargs):
        ac_na_a = self.forward(ob_no_a).reshape((-1,self.ac_dim))
        ac_na_b = ac_na_b.reshape((-1,self.ac_dim))
        input = torch.cat((ob_no_a,ob_no_b, ac_na_a , ac_na_b), dim = -1)
        loss = -torch.mean(critic.q_net_target(input))
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
            # c = torch.distributions.Categorical(logits=state)
            return state
        else:
            # c = torch.distributions.LowRankMultivariateNormal(self.mean_net(ptu.from_numpy(observation)).mul(ptu.from_numpy(self.env.action_space.high)),
            #                                                   10 ** self.cov_factor,
            #                                                   10 ** self.logstd)
            c = self.mean_net(observation)
            return c


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, discrete,lr,env,**kwargs):
        self.env = env
        super().__init__(ac_dim, ob_dim, n_layers, size, discrete,lr,**kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
        c = self.forward(observations.cpu())

        # log_p = c.log_prob(actions).reshape((-1, advantages.shape[-1]))
        log_p = c.log_prob(actions)
        # log_p = torch.sum(log_p, dim=1)
        log_p = log_p.reshape(advantages.shape)
        # q_val = (advantages.reshape((-1, advantages.shape[-1]))).clone().requires_grad_(True)
        q_val = advantages
        loss = torch.sum(torch.mul(log_p, q_val).mul(-1), -1)

        # TODO: optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            ## TODO: normalize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            targets = utils.normalize(q_values, np.mean(q_values), np.std(q_values))
            targets = ptu.from_numpy(targets)

            ## TODO: use the `forward` method of `self.baseline` to get baseline predictions
            baseline_predictions = torch.squeeze(self.baseline(observations))

            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            assert baseline_predictions.shape == targets.shape

            # TODO: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            # HINT: use `F.mse_loss`
            baseline_loss = F.mse_loss(targets, baseline_predictions)

            # TODO: optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            'Training Loss'  : ptu.to_numpy(loss)
        }
        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]
