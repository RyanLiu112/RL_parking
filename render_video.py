import argparse
import datetime
import os
import time

import gym
import parking_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import pybullet as p


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=False, help='render the environment')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--total_timesteps', type=int, default=int(2e6), help='total timesteps to run')
parser.add_argument('--save_freq', type=int, default=int(5e5), help='checkpoint save frequency')
parser.add_argument('--log_path', type=str, default='./log', help='logging path')
parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
parser.add_argument('--mode', type=str, default='2', choices=['1', '2', '3', '4', '5'], help='mode')

args = parser.parse_args()


# args.ckpt_path = 'log/DQN_1_0808_2123/dqn_agent.zip'
# args.ckpt_path = 'log/DQN_2_0809_1026/dqn_agent.zip'
# args.ckpt_path = 'log/DQN_3_0807_2313/dqn_agent.zip'
# args.ckpt_path = 'log/DQN_4_0809_0111/dqn_agent.zip'
args.ckpt_path = 'log/DQN_5_0808_2029/dqn_agent.zip'

args.mode = args.ckpt_path[8]

if args.mode in ['1', '2', '3']:
    cameraYaw = 0
else:
    cameraYaw = 180

# Evaluation
env = gym.make(args.env, render=True, mode=args.mode, render_video=True)
obs = env.reset()
model = DQN.load(args.ckpt_path, env=env, print_system_info=True)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"log/DQN_{args.mode}.mp4")
# log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"log/env_720.mp4")

episode_return = 0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    position, _ = p.getBasePositionAndOrientation(env.car)
    p.resetDebugVisualizerCamera(
        cameraDistance=5,
        cameraYaw=cameraYaw,
        cameraPitch=-40,
        cameraTargetPosition=position
    )
    time.sleep(1 / 240)

    episode_return += reward
    if done:
        break

# for cameraYaw in range(720):
#     position, _ = p.getBasePositionAndOrientation(env.car)
#     p.resetDebugVisualizerCamera(
#         cameraDistance=6,
#         cameraYaw=cameraYaw,
#         cameraPitch=-40,
#         cameraTargetPosition=position
#     )
#     time.sleep(1 / 240)

p.stopStateLogging(log_id)

env.close()
print(f'episode return: {episode_return}')
