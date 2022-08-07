import gym
import parking_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


ckpt = 'dqn_agent'

env = gym.make('parking_env-v0', render=False, multi_obs=False)
env.reset()

total_timesteps = int(5e6)
model = DQN('MlpPolicy', env, verbose=1, seed=0, tensorboard_log='log')
model.learn(total_timesteps=total_timesteps)
model.save(ckpt)
del model


# Evaluation
env = gym.make('parking_env-v0', render=True, multi_obs=False)
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
