# RL parking

基于DQN、A2C等算法实现模拟小车倒车入库，强化学习算法使用stable-baselines3实现，倒车环境



## Tasks

### 1. 垂直倒车入库

- 目标：小车从车位前出发，小车方向与车位方向对齐，实现垂直倒车入库，倒车过程中不能碰撞墙壁。
- 效果：

<img src="./imgs/DQN_1.gif" alt="DQN_1"  />



### 2. 侧方位-垂直倒车入库

- 目标：小车从环境中心点 [0, 0] 出发，车头方向与车位垂直，实现倒车入库，倒车过程中不能碰撞墙壁。
- 效果：

<img src="./imgs/DQN_2.gif" alt="DQN_2"  />



### 3. 侧方位-垂直倒车入库（车位旁边有小车）

- 目标：小车从环境中心点 [0, 0] 出发，车头方向与车位垂直，实现倒车入库，倒车过程中不能碰撞墙壁和车位旁的小车。
- 效果：

<img src="./imgs/DQN_3.gif" alt="DQN_3"  />



### 4. 侧方位-平行倒车入库

- 目标：小车从环境中心点 [0, 0] 出发，车头方向与车位平行，实现倒车入库，倒车过程中不能碰撞墙壁。
- 效果：

<img src="./imgs/DQN_4.gif" alt="DQN_4"  />



### 5. 斜方位-60度倒车入库

- 目标：小车从环境中心点 [0, 0] 出发，车头方向与车位成 60 度角，实现倒车入库，倒车过程中不能碰撞墙壁。
- 效果：

<img src="./imgs/DQN_5.gif" alt="DQN_5"  />



### 6. 任意位置出发倒车入库（车位旁边有小车）

- 目标：小车从环境任意点出发，车头方向随机，实现倒车入库，倒车过程中不能碰撞墙壁和车位旁的小车。
- 效果：



## Definition

### 1. Observation



### 2. Action



### 3. Reward



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

- 任务3：侧方位-垂直倒车入库（车位旁边有小车）

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

- 任务6：任意位置出发倒车入库（车位旁边有小车）

```
python dgn_agent.py --mode=6
```



## How to evaluate

```
python evaluate.py --mode=1
```





