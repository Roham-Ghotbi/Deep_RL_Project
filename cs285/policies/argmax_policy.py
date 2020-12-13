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

        actions = self.critic.q_net_target(observation).cpu()
        actions = torch.distributions.Categorical(logits=actions)
        # action = ptu.to_numpy(actions.argmax(dim=-1))
        action = ptu.to_numpy(actions.sample())
        return action.squeeze()