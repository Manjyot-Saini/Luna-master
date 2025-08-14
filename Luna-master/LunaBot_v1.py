import gym
import mujoco_py
import numpy as np
import os

class GregBot(gym.Env):
    def __init__(self, xml_path='car210.xml', simend=50):
        # Initialize MuJoCo data structures
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None  # Viewer for rendering
        self.simend = simend

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32)

        # Define the goal position (initially set to (1.0, 1.0))
        self.goal_position = np.array([1.0, 1.0])

    def reset(self):
        self.sim.reset()
        self._randomize_spawn()

    def step(self, action):
        # Take a step in the environment based on the given action
        self._apply_action(action)
        self.sim.step()
        observation = self._get_observation()

        # Calculate the distance to the goal
        distance_to_goal = np.linalg.norm(observation[:2] - self.goal_position)

        # Define a reward function based on the negative distance to the goal
        reward = -distance_to_goal

        done = self.sim.data.time >= self.simend
        info = {'distance_to_goal': distance_to_goal}

        return observation, reward, done, info

    def render(self, mode='human'):
        # Render the current state of the environment
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()

    def close(self):
        # Clean up resources when the environment is closed
        if self.viewer is not None:
            self.viewer = None
    
    # OSERVE ~ l00k around
    def get_goal_coord(self):
        x = self.sim.data.get_body_xpos("goal")
        y = self.sim.data.get_body_ypos("goal")
        return np.array([x,y])
    
    def set_goal_coord(self, x, y):
        goal_pos_addr = self.model.get_joint_qpos_addr("goal")
        self.sim.data.qpos[goal_pos_addr] = np.array([x,y])
    
    def get_car_coord(self):
        x = self.sim.data.get_body_xpos("car")
        y = self.sim.data.get_body_ypos("car")
        return np.array([x,y])
    
    def get_car_quat(self):
        quat_x = self.sim.data.get_body_xquat("car")
        quat_y = self.sim.data.get_body_yquat("car")
        quat_z = self.sim.data.get_body_zquat("car")
        quat_w = self.sim.data.get_body_wquat("car")
        return np.array([quat_x, quat_y, quat_z, quat_w])
    
    # DO SHIT ~ Take ACTion!!

    def move_car(self, w, v):
        qvel_addr = self.model.get_joint_qvel_addr('car')
        self.sim.data.qvel[qvel_addr] = v
        self.sim.data.qvel[qvel_addr+1] = w
        self.sim.step




    def _randomize_spawn(self):
        # Randomize the spawn location of the bot and the goal point
        # You can adjust the ranges based on your desired spawn locations
        bot_spawn = np.random.uniform(low=-1.0, high=1.0, size=(2,))
        goal_spawn = np.random.uniform(low=-1.0, high=1.0, size=(2,))

        # Update the bot's position
        self.sim.data.qpos[0] = bot_spawn[0]
        self.sim.data.qpos[1] = bot_spawn[1]

        # Randomize the bot's orientation (in radians)
        

        # Update the goal's position
        self.goal_position = goal_spawn

# Create and use the environment
env = GregBot()
observation = env.reset()

count = 0
for _ in range(100000000000):
    action = env.action_space.sample()
    action = [0, 0]
    observation, reward, done, info = env.step(action)
    env.render()

    print(env.sim.data.qpos)

    count+=1 #
    if (count%300 ==0):
        done = True

    if done:
        done = False
        observation = env.reset()

env.close()
