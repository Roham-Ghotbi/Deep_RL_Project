import numpy as np
import torch

class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        # TODO return the action that maximizes the Q-value
        # at the current observation as the output
        actions = self.critic.q_net_target(observation).cpu() # will be ac_dim x ac_dim
        action = np.array(torch.argmax(actions))
        return action.squeeze()