import gym
import mujoco_py
import numpy as np
import os


class GregBot(gym.Env):

    def __init__(self, xml_path='car210.xml', simend=50):
        # Initialize MuJoCo data structures
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None  # This that viewer thingy for rendering
        self.simend = simend

        # The action and observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32)

    def reset(self):
        self.sim.reset()
        return self.get_observation()

    def step(self, action):
        self.apply_action(action)
        self.sim.step()
        observation = self.get_observation()
        reward = 0.0  # You can define your own reward function
        done = self.sim.data.time >= self.simend
        info = {}  # Additional information if needed

        return observation, reward, done, info

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()

    def close(self):
        # Clean up resources if any...

        if self.viewer is not None:
            self.viewer = None

    def apply_action(self, action):
        self.sim.data.ctrl[:] = [action[0], action[1]]
        

    def get_observation(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel])
    
    def get_goal_pos(self):
        return self.sim.data.get_body_xpos("goal")
    
    def get_car_pos(self):
        return self.sim.data.get_body_xpos("car")
    
    def get_car_quat(self):
        return self.sim.data.get_body_xquat("car")
    
    

env = GregBot()
observation = env.reset()

for _ in range(10000000000000000000000):
    action = env.action_space.sample() 
    # action is basically v and w between -1 and 1
    action = np.array([0.01, 0])
    observation, reward, done, info = env.step(action)
    env.render()
    if _%100 == 0:
        print("Goal: \t", env.get_goal_pos())
        print("Car: \t", env.get_car_pos())
        print("Quat: \t", env.get_car_quat())
    if done:
        observation = env.reset()

env.close()
