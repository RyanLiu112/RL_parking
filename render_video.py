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

# time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d_%H%M')
# args.log_path = os.path.join(args.log_path, f'DQN_{args.mode}_{time}')
# args.ckpt_path = 'log/DQN_5_0808_1635/dqn_agent.zip'
args.ckpt_path = 'dqn_agent_2.zip'
# if not args.ckpt_path:
#     args.ckpt_path = os.path.join(args.log_path, f'dqn_agent')


# Evaluation
env = gym.make(args.env, render=True, mode=args.mode)
obs = env.reset()
model = DQN.load(args.ckpt_path, env=env, print_system_info=True)

log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"log/DQN_{args.mode}.mp4")

episode_return = 0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    position, _ = p.getBasePositionAndOrientation(env.car)
    p.resetDebugVisualizerCamera(
        cameraDistance=5,
        cameraYaw=0,
        cameraPitch=-40,
        cameraTargetPosition=position
    )
    time.sleep(1 / 20)

    episode_return += reward
    if done:
        break

p.stopStateLogging(log_id)

print(f'episode return: {episode_return}')
