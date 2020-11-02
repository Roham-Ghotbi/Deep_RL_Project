import numpy as np
import torch

from cs285.infrastructure import pytorch_util as ptu

class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if observation.shape[1:] != tuple([self.critic.ob_dim*2]):
            observation = torch.cat((observation,observation), dim = -1)
        # TODO return the action that maximizes the Q-value
        # at the current observation as the output
        actions = self.critic.q_net_target(observation).cpu() # will be ac_dim x ac_dim
        action = ptu.to_numpy(actions.argmax(dim=-1))
        return action.squeeze()