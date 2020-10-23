from .base_critic import BaseCritic
from torch import nn
from torch import optim
import torch
from cs285.infrastructure import pytorch_util as ptu


class DifferentialCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_network = ptu.build_mlp(
            self.ob_dim + self.ob_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, input):
        return self.critic_network(input).squeeze(1)

    def forward_np(self, input):
        obs = ptu.from_numpy(input)
        predictions = self(input)
        return ptu.to_numpy(predictions)

    def update(self, batch_a, batch_b):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        ob_no_a, ac_na_a, reward_n_a, next_ob_no_a, terminal_n_a = batch_a
        ob_no_b, ac_na_b, reward_n_b, next_ob_no_b, terminal_n_b = batch_b
        ob_no_a = ptu.from_numpy(ob_no_a)
        next_ob_no_a = ptu.from_numpy(next_ob_no_a)
        terminal_n_a = ptu.from_numpy(terminal_n_a)

        ob_no_b = ptu.from_numpy(ob_no_b)
        next_ob_no_b = ptu.from_numpy(next_ob_no_b)
        terminal_n_b = ptu.from_numpy(terminal_n_b)

        ob_no = torch.cat((ob_no_a, ob_no_b), dim=-1)
        next_ob_no = torch.cat((next_ob_no_a, next_ob_no_b), dim=-1)
        reward_n = ptu.from_numpy(reward_n_a - reward_n_b)

        terminal_n = torch.clamp(terminal_n_a+terminal_n_b,0,1)
        for j in range(self.num_target_updates):
            v_1 = self.forward(next_ob_no.to(ptu.device)).squeeze().detach()
            target = reward_n + self.gamma * v_1 * (1 - terminal_n)
            for i in range(self.num_grad_steps_per_target_update):
                loss = self.loss(target, self.forward(ob_no.to(ptu.device)))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



        return loss.item()
