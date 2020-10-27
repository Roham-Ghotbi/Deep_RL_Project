import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.policies.MLP_policy import MLPPolicy
from cs285.critics.dqn_critic import DQNCritic
import torch
from cs285.infrastructure import pytorch_util as ptu

class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']

        self.critic = DQNCritic(agent_params)
        # self.actor = ArgMaxPolicy(self.critic)
        self.actor = MLPPolicy(agent_params['ac_dim'],
                               agent_params['ob_dim'],
                               agent_params['n_layers'],
                               agent_params['size'],
                               agent_params['discrete'],
                               agent_params['learning_rate'],
                               env)
        lander = agent_params['env_name'].startswith('LunarLander')
        self.discrete = agent_params['discrete']
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs, self.agent_params['ac_dim'])
        eps = self.exploration.value(self.t)

        # TODO use epsilon greedy exploration when selecting action
        perform_random_action = np.random.random() < eps or self.t < self.learning_starts
        obv = self.last_obs
        if perform_random_action:
            # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            if self.discrete:
                action = int(np.random.rand() * self.num_actions)
            else:
                b = self.env.action_space.high
                a = self.env.action_space.low
                action = np.random.rand(self.num_actions) * (b-a) + a
        else:
            # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent 
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor. 
            obv = ptu.from_numpy(self.replay_buffer.encode_recent_observation())
            action = self.actor.get_action(obv).reshape((self.actor.ac_dim))
        
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        self.last_obs, reward, done, info = self.env.step(action)

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()
        return obv, [], action, reward, self.last_obs, done

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, batch_a, batch_b):
        # ob_no, ac_na, re_n, next_ob_no, terminal_n
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # TODO fill in the call to the update function using the appropriate tensors

            log_c = self.critic.update(batch_a, batch_b, self.actor)

            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
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
        self.t += 1
        return log
