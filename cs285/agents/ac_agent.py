from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.critics.diff_critic import DifferentialCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent

import torch
from cs285.infrastructure import pytorch_util as ptu


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            self.env
        )
        self.use_difference_critic = agent_params['use_difference_critic']
        if self.use_difference_critic:
            self.critic = DifferentialCritic(self.agent_params)
        else:
            self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, batch_a, batch_b):
        # TODO Implement the following pseudocode:
        if self.use_difference_critic:
            for i in range(self.agent_params['num_critic_updates_per_agent_update']):
                loss_c = self.critic.update(batch_a, batch_b)
        else:
            ob_no_a, ac_na_a, reward_n_a, next_ob_no_a, terminal_n_a = batch_a
            for i in range(self.agent_params['num_critic_updates_per_agent_update']):
                loss_c = self.critic.update(ob_no_a, ac_na_a, next_ob_no_a, reward_n_a, terminal_n_a)

        advantage = self.estimate_advantage(batch_a, batch_b)

        ob_no_a, ac_na_a, reward_n_a, next_ob_no_a, terminal_n_a = batch_a
        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            loss_a = self.actor.update(ob_no_a, ac_na_a, advantage)



        loss = OrderedDict()
        loss['Critic_Loss'] = loss_c
        loss['Actor_Loss'] = loss_a['Training Loss']

        return loss

    def estimate_advantage(self, batch_a, batch_b):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        ob_no_a, ac_na_a, reward_n_a, next_ob_no_a, terminal_n_a = batch_a
        ob_no_b, ac_na_b, reward_n_b, next_ob_no_b, terminal_n_b = batch_b
        ob_no = ptu.from_numpy(ob_no_a)
        next_ob_no = ptu.from_numpy(next_ob_no_a)
        terminal_n = ptu.from_numpy(terminal_n_a)

        ob_no_b = ptu.from_numpy(ob_no_b)
        next_ob_no_b = ptu.from_numpy(next_ob_no_b)
        terminal_n_b = ptu.from_numpy(terminal_n_b)

        reward_n = ptu.from_numpy(reward_n_a)

        if self.use_difference_critic:
            ob_no = torch.cat((ob_no, ob_no_b), dim=-1)
            next_ob_no = torch.cat((next_ob_no, next_ob_no_b), dim=-1)
            reward_n = reward_n - ptu.from_numpy(reward_n_b)
            terminal_n = torch.clamp(terminal_n + terminal_n_b,0,1)

        v_0 = self.critic.forward(ob_no)
        v_1 = self.critic.forward(next_ob_no)
        adv_n = ptu.to_numpy(reward_n + self.gamma * v_1* (1-terminal_n) - v_0)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size, random=True):
        if not random:
            return self.replay_buffer.sample_recent_data(batch_size)
        else:
            return self.replay_buffer.sample_random_data(batch_size)

