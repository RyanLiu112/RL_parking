import argparse
import datetime
import os
import time

import gym
import parking_env
from stable_baselines3 import DQN
import pybullet as p
import moviepy.editor as mpy


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=False, help='render the environment')
parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
parser.add_argument('--mode', type=str, default='1', choices=['1', '2', '3', '4', '5', '6'], help='mode')

args = parser.parse_args()


args.ckpt_path = 'log/DQN_1_0808_2123/dqn_agent.zip'
# args.ckpt_path = 'log/DQN_2_0809_1026/dqn_agent.zip'
# args.ckpt_path = 'log/DQN_3_0809_1123/dqn_agent.zip'
# args.ckpt_path = 'log/DQN_4_0809_0111/dqn_agent.zip'
# args.ckpt_path = 'log/DQN_5_0808_2029/dqn_agent.zip'
# args.ckpt_path = 'log/DQN_6_0809_1919/dqn_agent_5000000_steps.zip'

args.mode = args.ckpt_path[8]

if args.mode in ['1', '2', '3', '6']:
    cameraYaw = 0
else:
    cameraYaw = 180

# Evaluation
env = gym.make(args.env, render=True, mode=args.mode, render_video=True)
obs = env.reset()
model = DQN.load(args.ckpt_path, env=env, print_system_info=True)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

filepath = f"log/DQN_{args.mode}2.mp4"
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, filepath)
# log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"log/gif/DQN_{args.mode}.mp4")

episode_return = 0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    position, _ = p.getBasePositionAndOrientation(env.car)
    p.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=cameraYaw,
        cameraPitch=-40,
        cameraTargetPosition=position
    )
    time.sleep(1 / 240)

    episode_return += reward
    if done:
        break

# for i in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     position, _ = p.getBasePositionAndOrientation(env.car)
#     p.resetDebugVisualizerCamera(
#         cameraDistance=1.8,
#         cameraYaw=cameraYaw,
#         cameraPitch=-50,
#         cameraTargetPosition=position
#     )
#     time.sleep(1 / 240)
#
#     episode_return += reward
#     if done:
#         break

p.stopStateLogging(log_id)

env.close()
print(f'episode return: {episode_return}')

video = mpy.VideoFileClip(filepath).subclip(0.02)
video.write_videofile(filepath.replace('2.mp4', '.mp4'))
os.remove(filepath)
