import argparse
import datetime
import os.path

import gym
import parking_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=False, help='render the environment')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--total_timesteps', type=int, default=int(5e6), help='total timesteps to run')
parser.add_argument('--save_freq', type=int, default=int(1e6), help='checkpoint save frequency')
parser.add_argument('--log_path', type=str, default='./log', help='logging path')
parser.add_argument('--ckpt_path', type=str, default='dqn_agent', help='checkpoint path')
parser.add_argument('--mode', type=str, default='2', choices=['1', '2', '3', '4', '5'], help='mode')


args = parser.parse_args()
date = datetime.datetime.strftime(datetime.datetime.now(), '%m%d')
args.log_path = os.path.join(args.log_path, f'DQN_{args.mode}_{date}')
args.ckpt_path = os.path.join(args.log_path, f'DQN_{args.mode}_{date}/dqn_agent')

env = gym.make(args.env, render=args.render)
env.reset()

model = DQN('MlpPolicy', env, verbose=1, seed=args.seed)
logger = configure(args.log_path, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.log_path, name_prefix='rl_model')
model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
model.save(args.ckpt_path)
del model


# Evaluation
env = gym.make(args.env, render=True)
obs = env.reset()
model = DQN.load(args.ckpt_path, env=env, print_system_info=True)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(f'mean reward: {mean_reward}, std reward: {std_reward}')

episode_return = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_return += reward
    if done:
        break

print(f'episode return: {episode_return}')
