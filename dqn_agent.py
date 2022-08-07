import gym
import parking_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure


ckpt = 'dqn_agent'

env = gym.make('parking_env-v0', render=False)
env.reset()

log_path = './log'
total_timesteps = int(5e6)
model = DQN('MlpPolicy', env, verbose=1, seed=0)
logger = configure(log_path, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
checkpoint_callback = CheckpointCallback(save_freq=int(1e6), save_path=log_path, name_prefix='rl_model')
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
model.save(ckpt)
del model


# Evaluation
env = gym.make('parking_env-v0', render=True)
obs = env.reset()
model = DQN.load(ckpt, env=env, print_system_info=True)
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
