from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.utils import Path
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from cs285.agents.diffq_agent import DiffQAgent
from cs285.infrastructure.dqn_utils import (
        get_wrapper_by_name,
        register_custom_envs,
)

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 1
MAX_VIDEO_LEN = 400 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.save_vids = params['save_vid_rollout']
        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############
        register_custom_envs()


        self.env = gym.make(self.params['env_name'])


        self.env.seed(seed)
        # if 'env_wrappers' in self.params['agent_params']:
        #     # These operations are currently only for Atari envs
        #     self.env = wrappers.Monitor(self.env, os.path.join(self.params['logdir'], "gym"), force=True)
        #     self.env = params['agent_params']['env_wrappers'](self.env)

        # import plotting (locally if 'obstacles' env)

        import matplotlib
        matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete
        print('Environment is discrete: %s'%discrete)

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

        self.avg_eval_ret = []
        self.std_eval_ret = []

    def apply_noise(self, input, noise_ratio):
        n = np.random.normal(np.zeros(input.shape, dtype=np.float32), scale=np.abs(input) * noise_ratio)
        n = n.astype('float32')
        return input + n

    def run_training_loop(self, n_iter, collect_policy, eval_policy
                          ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = self.params['scalar_log_freq'] if isinstance(self.agent, DiffQAgent) else 1000
        paths = []
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False


            ob, ac, reward, next_ob, terminal = self.agent.step_env()

            ## applying noise to observations
            # ob = self.apply_noise(ob,self.noise_ratio)
            # next_ob = self.apply_noise(next_ob, self.noise_ratio)

            obs.append(ob)
            acs.append(ac)
            rewards.append(reward)
            next_obs.append(next_ob)
            terminals.append(terminal)
            if terminal:
                paths += [Path(obs, [], acs, rewards, next_obs, terminals)]
                obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
            envsteps_this_batch = 1
            train_video_paths = None

            self.total_envsteps += envsteps_this_batch

            all_logs = self.train_agent()

            if paths!=[] and (self.logvideo or self.logmetrics):
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            batch_a = self.agent.sample(self.params['train_batch_size'])
            batch_b = self.agent.sample(self.params['train_batch_size'])

            train_log = self.agent.train(batch_a, batch_b)
            all_logs.append(train_log)
        self.agent.update_t()
        return all_logs

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'], render=False)

        if self.logvideo: # and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            if self.save_vids:
                print('\nSaving eval rollouts as videos...')
                self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                                 video_title='eval_rollouts')
        #######################

        # save eval metrics
        if self.logmetrics :
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [path["reward"].size for path in paths]
            eval_ep_lens = [eval_path["reward"].size for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            self.avg_eval_ret += [np.mean(eval_returns)]

            logs["Eval_StdReturn"] = np.std(eval_returns)
            self.std_eval_ret += [np.std(eval_returns)]

            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
