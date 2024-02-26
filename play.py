from torch import device
from mujoco_py.generated import const
from mujoco_py import GlfwContext
import numpy as np
import cv2
#import time
GlfwContext(offscreen=True)


class Play:
    def __init__(self, env, agent, env_name, max_episode=1):
        self.env = env
        self.max_episode = max_episode
        self.agent = agent
        _, self.state_rms_mean, self.state_rms_var = self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cpu")

        
        

    def evaluate(self,k):

        for _ in range(self.max_episode):
            s = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.env._max_episode_steps):
                s = np.clip((s - self.state_rms_mean) / (self.state_rms_var ** 0.5 + 1e-8), -5.0, 5.0)
                dist = self.agent.choose_dist(s)
                action = dist.sample().cpu().numpy()[0]*k
                s_, r, done, _ = self.env.step(action)
 
                episode_reward += r
                if done:
                    break
                s = s_
            return episode_reward
        self.env.close()

