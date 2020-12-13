import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.policies.ActorPolicy import ActorPolicy
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.critics.diffq_critic import DiffQCritic
from cs285.critics.diffq_critic_base import DiffQCriticBase
from cs285.exploration.rnd_model import RNDModel

import torch
from cs285.infrastructure import pytorch_util as ptu

class DiffQAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.noise_ratio = self.agent_params['observation_noise_multiple']
        self.last_obs = self.apply_noise(self.env.reset(),self.noise_ratio)

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']


        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']

        self.critic = DiffQCritic(agent_params) if not agent_params['diff_training_disabled'] else DiffQCriticBase(agent_params)
        self.actor = ActorPolicy(agent_params['ac_dim'],
                                 agent_params['ob_dim'],
                                 agent_params['n_layers'],
                                 agent_params['size'],
                                 agent_params['discrete'],
                                 agent_params['learning_rate'],
                                 env,
                                 agent_params,
                                 self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.discrete = agent_params['discrete']
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0


    def add_to_replay_buffer(self, paths):
        pass

    def apply_noise(self, input, noise_ratio):
        b = self.env.action_space.high
        a = self.env.action_space.low

        # n = np.random.normal(np.zeros(input.shape, dtype=np.float32), scale=np.sqrt(np.square(input)) * noise_ratio)
        n = np.random.normal(np.zeros(input.shape, dtype=np.float32), scale=b * noise_ratio)
        n = n.astype('float32')
        return input + n


    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """

        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs, self.agent_params['ac_dim'])
        eps = self.exploration.value(self.t)

        perform_random_action = np.random.random() < eps or self.t < self.learning_starts
        obv = self.last_obs
        if perform_random_action:
            if self.discrete:
                action = int(np.random.rand() * self.num_actions)
            else:
                b = self.env.action_space.high
                a = self.env.action_space.low
                action = np.random.rand(self.num_actions) * (b-a) + a
        else:
            action = self.actor.get_action(ptu.from_numpy(obv)).reshape((self.actor.ac_dim) if not self.discrete else -1)
            if self.discrete:
                action = action.item()
        
        self.last_obs, reward, done, info = self.env.step(action)

        self.last_obs = self.apply_noise(self.last_obs, self.noise_ratio)

        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.apply_noise(self.env.reset(), self.noise_ratio)
        return obv, action, reward, self.last_obs, done

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, batch_a, batch_b):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            log_c = self.critic.update(batch_a, batch_b, self.actor)

            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()


            ob_no_a, ac_na_a, _, _, _ = batch_a
            ob_no_a = ptu.from_numpy(ob_no_a)
            ac_na_a = ptu.from_numpy(ac_na_a)

            ob_no_b, ac_na_b, _, _, _ = batch_b
            ob_no_b = ptu.from_numpy(ob_no_b)
            ac_na_b = ptu.from_numpy(ac_na_b)

            log_a = self.actor.update(ob_no_a, ob_no_b, ac_na_b, self.critic)
            self.num_param_updates += 1
            log = {'Critic Loss': log_c['Critic Loss'],
                   'Actor Loss': log_a['Actor Loss']}
        return log

    def update_t(self):
        self.t += 1
