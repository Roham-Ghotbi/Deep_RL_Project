import os
import time, glob
import scipy.io as sio

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.diffq_agent import DiffQAgent
from cs285.infrastructure.dqn_utils import get_env_kwargs


class diffQ_Trainer(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
        }

        env_args = get_env_kwargs(params['env_name'])

        self.agent_params = {**train_args, **env_args, **params}

        self.params['agent_class'] = DiffQAgent
        self.agent_params['gamma'] = params['discount']
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            # self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
        )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        default='MsPacman-v0',
        # choices=('PongNoFrameskip-v4', 'LunarLander-v3','LunarLander-v2', 'MsPacman-v0','InvertedPendulum-v2',
        #          'LunarLanderContinuous-v2','Pendulum-v0')
    )

    parser.add_argument('--ep_len', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=4000)

    parser.add_argument('--batch_size', '-b', type=int, default=16)  # steps collected per train iteration

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=3)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--double_q', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1000))
    parser.add_argument('--video_log_freq', type=int, default=10000)

    parser.add_argument('--observation_noise_multiple', type=float, default=0.15)
    parser.add_argument('--action_noise_multiple', type=float, default=0.)

    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=100)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--save_vid_rollout', action='store_true', default=False)
    
    parser.add_argument('--learning_rate','-lr', type=float, default=1)
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = diffQ_Trainer(params)
    trainer.run_training_loop()
    # t = (glob.glob('*.mat')).__len__()
    # sio.savemat('%s.mat'%(params['exp_name']), {'avg_eval_ret': trainer.rl_trainer.avg_eval_ret, 'std_eval_ret': trainer.rl_trainer.std_eval_ret})
    # sio.savemat(os.path.dirname(os.path.realpath(__file__))+'%s_%d.mat' % (params['exp_name'],params['batch_size']), {'avg_eval_ret': trainer.rl_trainer.avg_eval_ret, 'std_eval_ret': trainer.rl_trainer.std_eval_ret})


if __name__ == "__main__":
    main()
