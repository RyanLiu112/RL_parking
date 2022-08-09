# RL parking

基于DQN、A2C等算法实现模拟小车倒车入库，强化学习算法使用stable-baselines3实现，倒车环境



## Tasks

- 任务1：垂直倒车入库

![DQN_1](/Users/liurunze/RL_parking/imgs/DQN_1.gif)



- 任务2：侧方位-垂直倒车入库

![DQN_2](/Users/liurunze/RL_parking/imgs/DQN_2.gif)



- 任务3：侧方位-垂直倒车入库（旁边有小车）

![DQN_3](/Users/liurunze/RL_parking/imgs/DQN_3.gif)



- 任务4：侧方位-平行倒车入库

![DQN_4](/Users/liurunze/RL_parking/imgs/DQN_4.gif)



- 任务5：斜方位-30度倒车入库

![DQN_5](/Users/liurunze/RL_parking/imgs/DQN_5.gif)



## How to install

- 克隆本项目

```
git clone https://github.com/RyanLiu112/RL_parking
```

- 进入项目目录

```
cd RL_parking
```

- 安装停车环境（会同时安装gym、pybullet、stable-baselines3等第三方库）

```
pip install -e parking_env
```



## How to play in the environment

```
python play.py --mode=1
```



## How to train

### 1. 使用DQN算法训练小车停车智能体

- 任务1：垂直倒车入库

```
python dqn_agent.py --mode=1
```

- 任务2：侧方位-垂直倒车入库

```
python dqn_agent.py --mode=2
```

- 任务3：侧方位-垂直倒车入库（旁边有小车）

```
python dqn_agent.py --mode=3
```

- 任务4：侧方位-平行倒车入库

```
python dqn_agent.py --mode=4
```

- 任务5：斜方位-30度倒车入库

```
python dqn_agent.py --mode=5
```



## How to evaluate

```
python evaluate.py --mode=1
```



