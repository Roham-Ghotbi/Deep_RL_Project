from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from cs285.infrastructure import pytorch_util as ptu
global device

class DQNCritic(BaseCritic):

    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim + self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.gamma = hparams['gamma']
        self.n_layers = hparams['n_layers']
        self.size = hparams['size']
        self.discrete = hparams['discrete']
        self.grad_norm_clipping = hparams['grad_norm_clipping']

        if not self.discrete:
            self.q_net = ptu.build_mlp(
            self.ob_dim + self.ob_dim + self.ac_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size
            )
            self.q_net_target = ptu.build_mlp(
                self.ob_dim + self.ob_dim + self.ac_dim + self.ac_dim,
                1,
                n_layers=self.n_layers,
                size=self.size
            )
            self.loss = nn.MSELoss()  # AKA Huber loss
        else:
            self.q_net = ptu.build_mlp(
                self.ob_dim + self.ob_dim + 1,
                self.ac_dim,
                n_layers=self.n_layers,
                size=self.size
            )
            self.q_net_target = ptu.build_mlp(
                self.ob_dim + self.ob_dim + 1,
                self.ac_dim,
                n_layers=self.n_layers,
                size=self.size
            )
            self.loss = nn.MSELoss()  # AKA Huber loss
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            hparams['learning_rate']
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
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

    def update(self, batch_a, batch_b, actor):
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
                nothing
        """
        # batch a
        ob_no_a, ac_na_a, reward_n_a, next_ob_no_a, terminal_n_a = self.unwrap_path(batch_a)
        ac_tp1_a = ptu.from_numpy(actor.get_action(next_ob_no_a)).reshape((ac_na_a.shape[0],-1)) # if not self.discrete else actor.get_action(next_ob_no_a).unsqueeze(-1).to(dtype=torch.float32)

        # batch b
        ob_no_b, ac_na_b, reward_n_b, next_ob_no_b,  terminal_n_b = self.unwrap_path(batch_b)
        ac_tp1_b = ptu.from_numpy(actor.get_action(next_ob_no_b)).reshape((ac_na_a.shape[0],-1)) # if not self.discrete else actor(next_ob_no_a).get_action.unsqueeze(-1).to(dtype=torch.float32)

        terminal_n = torch.clamp(terminal_n_a + terminal_n_b, 0, 1)
        next_ob_no = torch.cat((next_ob_no_a, next_ob_no_b), dim = 1)
        ob_no = torch.cat((ob_no_a, ob_no_b), dim = 1)
        ac_tp1 = torch.cat((ac_tp1_a, ac_tp1_b), dim = 1)
        ac_na = torch.cat((ac_na_a.unsqueeze(-1), ac_na_b.unsqueeze(-1)), dim = 1).squeeze(-1)
        reward_n = reward_n_a - reward_n_b


        if self.discrete:
            input = torch.cat((ob_no, ac_na_b), dim=1)
            diff_Qs = self.q_net(input.to(ptu.device))
            diff_Qs = torch.gather(diff_Qs, 1, ac_na_a.to(dtype=torch.int64).detach()).squeeze(1)
        else:
            input = torch.cat((ob_no, ac_na), dim = 1)
            diff_Qs = self.q_net(input.to(ptu.device)).squeeze(-1)

        if self.double_q:
            # You must fill this part for Q2 of the Q-learning portion of the homework.
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network. See page 5 of https://arxiv.org/pdf/1509.06461.pdf for more details.
            qa_tp1_values = self.q_net(next_ob_no.to(ptu.device))
            _ , ind = qa_tp1_values.max(dim=1) # need to be changed
            qa_tp1_values = self.q_net_target(next_ob_no.to(ptu.device)).reshape((-1,self.ac_dim,self.ac_dim))
            diff_Qs_tp1 = torch.gather(qa_tp1_values, 1, ind.unsqueeze(1)).squeeze(1)
        else:
            if self.discrete:
                input_tp1 = torch.cat((next_ob_no, ac_tp1_b), dim=-1)
                diff_Qs_tp1 = self.q_net_target(input_tp1.to(ptu.device))
                diff_Qs_tp1 = torch.gather(diff_Qs_tp1, 1, ac_na_a.to(dtype=torch.int64).detach()).squeeze(1)
            else:
                input_tp1 = torch.cat((next_ob_no, ac_tp1), dim = -1)
                diff_Qs_tp1 = self.q_net_target(input_tp1.to(ptu.device))

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma * diff_Qs_tp1.squeeze(-1) #* (1-terminal_n)
        target = target.detach()


        assert diff_Qs.shape == target.shape
        loss = self.loss(diff_Qs, target)

        self.optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()


        return {
            'Critic Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def unwrap_path(self, batch):
        ob_no, ac_na, reward_n, next_ob_no, terminal_n = batch
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        if self.discrete:
            ac_na = ac_na[:,0].unsqueeze(1)
        return ob_no, ac_na, reward_n, next_ob_no, terminal_n

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
