import os
import random
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces


class CustomEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, base_path=os.getcwd(), car_type='husky', mode='1', manual=False, multi_obs=False, render_video=False):
        """
        初始化环境

        :param render: 是否渲染GUI界面
        :param base_path: 项目路径
        :param car_type: 小车类型（husky）
        :param mode: 任务类型
        :param manual: 是否手动操作
        :param multi_obs: 是否使用多个observation
        :param render_video: 是否渲染视频
        """

        self.base_path = base_path
        self.car_type = car_type
        self.manual = manual
        self.multi_obs = multi_obs
        self.mode = mode
        assert self.mode in ['1', '2', '3', '4', '5', '6']

        self.car = None
        self.done = False
        self.goal = None
        self.desired_goal = None

        self.ground = None
        self.left_wall = None
        self.right_wall = None
        self.front_wall = None
        self.parked_car1 = None
        self.parked_car2 = None

        # 定义状态空间
        obs_low = np.array([0, 0, -1, -1, -1, -1])
        obs_high = np.array([20, 20, 1, 1, 1, 1])
        if multi_obs:
            self.observation_space = spaces.Dict(
                spaces={
                    "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "achieved_goal": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "desired_goal": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                }
            )
        else:
            self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 4种动作：前进、后退、左转、右转

        # self.reward_weights = np.array([1, 0.3, 0, 0, 0.02, 0.02])
        if self.mode == '4':
            self.reward_weights = np.array([0.4, 0.9, 0, 0, 0.1, 0.1])
        elif self.mode == '5':
            self.reward_weights = np.array([0.65, 0.65, 0, 0, 0.1, 0.1])
        else:
            self.reward_weights = np.array([1, 0.3, 0, 0, 0.1, 0.1])
        self.target_orientation = None
        self.start_orientation = None

        if self.mode in ['1', '2', '3', '5', '6']:
            self.action_steps = 5
        else:
            self.action_steps = 3
        self.step_cnt = 0
        self.step_threshold = 500

        if render:
            self.client = p.connect(p.GUI)
            time.sleep(1. / 240.)
        else:
            self.client = p.connect(p.DIRECT)
            time.sleep(1. / 240.)
        if render and render_video:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

    def render(self, mode='human'):
        """
        渲染当前画面

        :param mode: 渲染模式
        """

        p.stepSimulation(self.client)
        time.sleep(1. / 240.)

    def reset(self):
        """
        重置环境

        """

        # reset function to be made for muliple-agents
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # new Loading the plane
        self.ground = p.loadURDF(os.path.join(self.base_path, "assets/arena_new.urdf"), basePosition=[0, 0, 0.005], useFixedBase=10)

        p.addUserDebugLine([-3.5, -3.5, 0.02], [-3.5, 3.5, 0.02], [0.75, 0.75, 0.75], 5)
        p.addUserDebugLine([-3.5, -3.5, 0.02], [3.5, -3.5, 0.02], [0.75, 0.75, 0.75], 5)
        p.addUserDebugLine([3.5, 3.5, 0.02], [3.5, -3.5, 0.02], [0.75, 0.75, 0.75], 5)
        p.addUserDebugLine([3.5, 3.5, 0.02], [-3.5, 3.5, 0.02], [0.75, 0.75, 0.75], 5)

        # mode = 1, 2 (右上)
        if self.mode == '1' or self.mode == '2':
            self.left_wall = p.loadURDF(os.path.join(self.base_path, "assets/up/side_boundary.urdf"), basePosition=[1.3, 2.1, 0.03], useFixedBase=10)
            self.right_wall = p.loadURDF(os.path.join(self.base_path, "assets/up/side_boundary.urdf"), basePosition=[2.5, 2.1, 0.03], useFixedBase=10)
            self.front_wall = p.loadURDF(os.path.join(self.base_path, "assets/up/front_boundary_ru.urdf"), basePosition=[1.9, 2.8, 0.03], useFixedBase=10)
        # else:
        #     p.loadURDF(os.path.join(self.base_path, "assets/up/side_boundary.urdf"), basePosition=[1.3, 2.1, 0.03], useFixedBase=10)
        #     p.loadURDF(os.path.join(self.base_path, "assets/up/side_boundary.urdf"), basePosition=[2.5, 2.1, 0.03], useFixedBase=10)
        #     p.loadURDF(os.path.join(self.base_path, "assets/up/front_boundary_ru.urdf"), basePosition=[1.9, 2.8, 0.03], useFixedBase=10)
        # p.addUserDebugLine([1.4, 1.5, 0.02], [1.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([1.4, 1.5, 0.02], [2.4, 1.5, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([2.4, 2.7, 0.02], [1.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([2.4, 2.7, 0.02], [2.4, 1.5, 0.02], [0.98, 0.98, 0.98], 2.5)

        # mode = 3, 6 (左上)
        if self.mode == '3' or self.mode == '6':
            self.left_wall = p.loadURDF(os.path.join(self.base_path, "assets/up/side_boundary.urdf"), basePosition=[-0.3, 2.1, 0.03], useFixedBase=10)
            self.right_wall = p.loadURDF(os.path.join(self.base_path, "assets/up/side_boundary.urdf"), basePosition=[-3.5, 2.1, 0.03], useFixedBase=10)
            self.front_wall = p.loadURDF(os.path.join(self.base_path, "assets/up/front_boundary_lu.urdf"), basePosition=[-1.9, 2.8, 0.03], useFixedBase=10)
            self.parked_car1 = p.loadURDF("husky/husky.urdf", basePosition=[-0.9, 2.1, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)
            self.parked_car2 = p.loadURDF("husky/husky.urdf", basePosition=[-2.9, 2.1, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)
        # else:
        #     p.loadURDF(os.path.join(self.base_path, "assets/up/side_boundary.urdf"), basePosition=[-0.3, 2.1, 0.03], useFixedBase=10)
        #     p.loadURDF(os.path.join(self.base_path, "assets/up/side_boundary.urdf"), basePosition=[-3.5, 2.1, 0.03], useFixedBase=10)
        #     p.loadURDF(os.path.join(self.base_path, "assets/up/front_boundary_lu.urdf"), basePosition=[-1.9, 2.8, 0.03], useFixedBase=10)
        #     p.loadURDF("husky/husky.urdf", basePosition=[-0.9, 2.1, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)
        #     p.loadURDF("husky/husky.urdf", basePosition=[-2.9, 2.1, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)
        # p.addUserDebugLine([-0.4, 1.5, 0.02], [-3.4, 1.5, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-0.4, 2.7, 0.02], [-3.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-0.4, 1.5, 0.02], [-0.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-1.4, 1.5, 0.02], [-1.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-2.4, 1.5, 0.02], [-2.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-3.4, 1.5, 0.02], [-3.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)

        # mode = 4 (左下)
        if self.mode == '4':
            self.left_wall = p.loadURDF(os.path.join(self.base_path, "assets/down/side_boundary_ld.urdf"), basePosition=[-0.8, -2.1, 0.03], useFixedBase=10)
            self.right_wall = p.loadURDF(os.path.join(self.base_path, "assets/down/side_boundary_ld.urdf"), basePosition=[-3.0, -2.1, 0.03], useFixedBase=10)
            self.front_wall = p.loadURDF(os.path.join(self.base_path, "assets/down/front_boundary_ld.urdf"), basePosition=[-1.9, -2.7, 0.03], useFixedBase=10)
        # else:
        #     p.loadURDF(os.path.join(self.base_path, "assets/down/side_boundary_ld.urdf"), basePosition=[-0.8, -2.1, 0.03], useFixedBase=10)
        #     p.loadURDF(os.path.join(self.base_path, "assets/down/side_boundary_ld.urdf"), basePosition=[-3.0, -2.1, 0.03], useFixedBase=10)
        #     p.loadURDF(os.path.join(self.base_path, "assets/down/front_boundary_ld.urdf"), basePosition=[-1.9, -2.7, 0.03], useFixedBase=10)
        # p.addUserDebugLine([-0.9, -1.6, 0.02], [-2.9, -1.6, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-0.9, -2.6, 0.02], [-2.9, -2.6, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-0.9, -1.6, 0.02], [-0.9, -2.6, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-2.9, -1.6, 0.02], [-2.9, -2.6, 0.02], [0.98, 0.98, 0.98], 2.5)

        # mode = 5 (右下)
        if self.mode == '5':
            self.left_wall = p.loadURDF(os.path.join(self.base_path, "assets/down/side_boundary_rd.urdf"), basePosition=[1.6, -2.15, 0.03], useFixedBase=10)
            self.right_wall = p.loadURDF(os.path.join(self.base_path, "assets/down/side_boundary_rd.urdf"), basePosition=[2.9, -2.15, 0.03], useFixedBase=10)
            self.front_wall = p.loadURDF(os.path.join(self.base_path, "assets/down/front_boundary_rd.urdf"), basePosition=[2.55, -2.8, 0.03], useFixedBase=10)
        # else:
        #     p.loadURDF(os.path.join(self.base_path, "assets/down/side_boundary_rd.urdf"), basePosition=[1.6, -2.15, 0.03], useFixedBase=10)
        #     p.loadURDF(os.path.join(self.base_path, "assets/down/side_boundary_rd.urdf"), basePosition=[2.9, -2.15, 0.03], useFixedBase=10)
        #     p.loadURDF(os.path.join(self.base_path, "assets/down/front_boundary_rd.urdf"), basePosition=[2.55, -2.8, 0.03], useFixedBase=10)
        # p.addUserDebugLine([1.4, -1.6, 0.02], [2.5, -1.6, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([2.0, -2.7, 0.02], [3.1, -2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([1.4, -1.6, 0.02], [2.0, -2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([2.5, -1.6, 0.02], [3.1, -2.7, 0.02], [0.98, 0.98, 0.98], 2.5)

        # 起始点
        # p.addUserDebugLine([-0.2, -0.2, 0.02], [-0.2, 0.2, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([-0.2, -0.2, 0.02], [0.2, -0.2, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([0.2, 0.2, 0.02], [-0.2, 0.2, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([0.2, 0.2, 0.02], [0.2, -0.2, 0.02], [0.98, 0.98, 0.98], 2.5)

        basePosition = [0, 0, 0.2]
        if self.mode == '1':
            self.goal = np.array([3.8 / 2, 4.2 / 2])
            self.start_orientation = [0, 0, np.pi * 3 / 2]
            self.target_orientation = np.pi * 3 / 2
            basePosition = [1.9, -0.2, 0.2]
        elif self.mode == '2':
            self.goal = np.array([3.8 / 2, 4.2 / 2])
            self.start_orientation = [0, 0, np.pi * 2 / 2]
            self.target_orientation = np.pi * 3 / 2
        elif self.mode == '3':
            self.goal = np.array([-3.8 / 2, 4.2 / 2])
            self.start_orientation = [0, 0, np.pi * 4 / 2]
            self.target_orientation = np.pi * 3 / 2
        elif self.mode == '4':
            self.goal = np.array([-3.8 / 2, -4.2 / 2])
            self.start_orientation = [0, 0, np.pi * 0 / 2]
            self.target_orientation = np.pi * 0 / 2
        elif self.mode == '5':
            self.goal = np.array([4.53 / 2, -4.55 / 2])
            self.start_orientation = [0, 0, np.pi * 2 / 2]
            self.target_orientation = 2.070143
        elif self.mode == '6':
            self.goal = np.array([-3.8 / 2, 4.2 / 2])
            self.start_orientation = [0, 0, np.random.rand() * 2 * np.pi]
            self.target_orientation = np.pi * 3 / 2
            while 1:
                random_x = (np.random.rand() - 0.5) * 4
                random_y = (np.random.rand() - 0.5) * 4
                if random_x < -1 and random_y > 1:
                    continue
                else:
                    break
            basePosition = [random_x, random_y, 0.2]

        self.desired_goal = np.array([self.goal[0], self.goal[1], 0.0, 0.0, np.cos(self.target_orientation), np.sin(self.target_orientation)])

        # 加载小车
        # basePosition = [np.random.rand() * 3 + 2, np.random.rand() * 8 + 1, 0.2]
        self.t = Car(self.client, basePosition=basePosition, baseOrientationEuler=self.start_orientation, carType=self.car_type, action_steps=self.action_steps)
        self.car = self.t.car

        # 获取当前observation
        car_ob, self.vector = self.t.get_observation()
        observation = np.array(list(car_ob))

        self.step_cnt = 0

        if self.multi_obs:
            observation = {
                'observation': observation,
                'achieved_goal': observation,
                'desired_goal': self.desired_goal
            }

        return observation

    def distance_function(self, pos):
        """
        计算小车与目标点的距离（2-范数）

        :param pos: 小车当前坐标 [x, y, z]
        :return: 小车与目标点的距离
        """

        return np.sqrt(pow(pos[0] - self.goal[0], 2) + pow(pos[1] - self.goal[1], 2))

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        计算当前步的奖励

        :param achieved_goal: 小车当前位置 [x, y, z]
        :param desired_goal: 目标点 [x, y, z]
        :param info: 信息
        :return: 奖励
        """

        p_norm = 0.5
        reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.reward_weights)), p_norm)

        return reward

    def judge_collision(self):
        """
        判断小车与墙壁、停放着的小车是否碰撞

        :return: 是否碰撞
        """

        done = False
        points1 = p.getContactPoints(self.car, self.left_wall)
        points2 = p.getContactPoints(self.car, self.right_wall)
        points3 = p.getContactPoints(self.car, self.front_wall)

        if len(points1) or len(points2) or len(points3):
            done = True
        if self.mode == '3':
            points4 = p.getContactPoints(self.car, self.parked_car1)
            points5 = p.getContactPoints(self.car, self.parked_car2)
            if len(points4) or len(points5):
                done = True

        return done

    def step(self, action):
        """
        环境步进

        :param action: 小车动作
        :return: observation, reward, done, info
        """

        self.t.apply_action(action)  # 小车执行动作
        p.stepSimulation()
        car_ob, self.vector = self.t.get_observation()  # 获取小车状态

        position = np.array(car_ob[:2])
        distance = self.distance_function(position)
        reward = self.compute_reward(car_ob, self.desired_goal, None)

        if self.manual:
            print(f'dis: {distance}, reward: {reward}, center: {self.goal}, pos: {car_ob}')

        self.done = False
        self.success = False

        if distance < 0.1:
            self.success = True
            self.done = True

        self.step_cnt += 1
        if self.step_cnt > self.step_threshold:  # 限制episode长度为step_threshold
            self.done = True
        if car_ob[2] < -2:  # 小车掉出环境
            # print('done! out')
            reward = -500
            self.done = True
        if self.judge_collision():  # 碰撞
            # print('done! collision')
            reward = -500
            self.done = True
        if self.done:
            self.step_cnt = 0

        observation = np.array(list(car_ob))
        if self.multi_obs:
            observation = {
                'observation': observation,
                'achieved_goal': observation,
                'desired_goal': self.desired_goal
            }

        info = {'is_success': self.success}

        return observation, reward, self.done, info

    def seed(self, seed=None):
        """
        设置环境种子

        :param seed: 种子
        :return: [seed]
        """

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        """
        关闭环境

        """

        p.disconnect(self.client)


class Car:
    def __init__(self, client, basePosition=[0, 0, 0.2], baseOrientationEuler=[0, 0, np.pi / 2],
                 max_velocity=6, max_force=100, carType='husky', action_steps=None):
        """
        初始化小车

        :param client: pybullet client
        :param basePosition: 小车初始位置
        :param baseOrientationEuler: 小车初始方向
        :param max_velocity: 最大速度
        :param max_force: 最大力
        :param carType: 小车类型
        :param action_steps: 动作步数
        """

        self.client = client
        urdfname = carType + '/' + carType + '.urdf'
        self.car = p.loadURDF(fileName=urdfname, basePosition=basePosition, baseOrientation=p.getQuaternionFromEuler(baseOrientationEuler))

        self.steering_joints = [0, 2]
        self.drive_joints = [1, 3, 4, 5]

        self.max_velocity = max_velocity
        self.max_force = max_force
        self.action_steps = action_steps

    def apply_action(self, action):
        """
        小车执行动作

        :param action: 动作
        """

        velocity = self.max_velocity  # rad/s
        force = self.max_force  # Newton

        if action == 0:  # forward
            for i in range(self.action_steps):
                for joint in range(2, 6):
                    p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL, targetVelocity=velocity, force=force)
                p.stepSimulation()
        elif action == 1:  # back
            for i in range(self.action_steps):
                for joint in range(2, 6):
                    p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL, targetVelocity=-velocity, force=force)
                p.stepSimulation()
        elif action == 2:  # left
            targetVel = 3
            for i in range(self.action_steps):
                for joint in range(2, 6):
                    for joint in range(1, 3):
                        p.setJointMotorControl2(self.car, 2 * joint + 1, p.VELOCITY_CONTROL,
                                                targetVelocity=targetVel, force=force)
                    for joint in range(1, 3):
                        p.setJointMotorControl2(self.car, 2 * joint, p.VELOCITY_CONTROL, targetVelocity=-targetVel,
                                                force=force)
                    p.stepSimulation()
        elif action == 3:  # right
            targetVel = 3
            for i in range(self.action_steps):
                for joint in range(2, 6):
                    for joint in range(1, 3):
                        p.setJointMotorControl2(self.car, 2 * joint, p.VELOCITY_CONTROL, targetVelocity=targetVel,
                                                force=force)
                    for joint in range(1, 3):
                        p.setJointMotorControl2(self.car, 2 * joint + 1, p.VELOCITY_CONTROL,
                                                targetVelocity=-targetVel, force=force)
                    p.stepSimulation()
        elif action == 4:  # stop
            targetVel = 0
            for joint in range(2, 6):
                p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL, targetVelocity=targetVel,
                                        force=force)
            p.stepSimulation()
        else:
            raise ValueError

    def get_observation(self):
        """
        获取小车当前状态

        :return: observation, vector
        """

        position, angle = p.getBasePositionAndOrientation(self.car)  # 获取小车位姿
        angle = p.getEulerFromQuaternion(angle)
        velocity = p.getBaseVelocity(self.car)[0]

        position = [position[0], position[1]]
        velocity = [velocity[0], velocity[1]]
        orientation = [np.cos(angle[2]), np.sin(angle[2])]
        vector = angle[2]

        observation = np.array(position + velocity + orientation)  # 拼接坐标、速度、角度

        return observation, vector
