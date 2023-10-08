import pyglet
from threading import Thread
from time import sleep
import numpy as np
from pyglet import gl
from gym.spaces import Box,Discrete,Tuple
from pyglet.window import Window


# state of agents (including communication and internal/mental state)
class AgentState():
    def __init__(self):
        super(AgentState, self).__init__()
        self.p_posx = None
        self.p_posy = None
        self.p_vel = None
        self.p_direct = None

        # 1表示进行诱骗
        self.lure = None
        self.disturb = None


# action of the agent
class Action(object):
    def __init__(self):
        self.acclr = None
        self.angle_change = None

        # d_act仅仅对于有这个动作空间的才能用
        # 1表示进行干扰或者诱骗、0表示不进行
        self.d_disturb = None
        self.d_lure = None


class Agent():
    # maxspeed单位是节  accel单位是m/s^2
    def __init__(self):
        super(Agent, self).__init__()
        self.movable = True
        self.min_speed = None
        self.max_speed = None
        self.accel = None
        self.size = None
        self.color = None

        self.obs_dim = None

        self.is_lure = False
        self.is_disturb = False

        self.lured = False

        self.state = AgentState()
        self.action = Action()

        self.action_callback = None


class Environment:

    def __init__(self, num_agents=4):
        self.num_agents = num_agents

        self.observation_space = []
        self.action_space = []
        self.action_space = []

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)
        # 蓝A，红A，红B1(诱骗)，红B2(干扰)最大速度分别为：40节、30节、18节、18节
        speed_list = [40, 30, 18, 18]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)最大加速度分别为：0,0.02,0.02,0.02
        accel_list = [0, 0.02, 0.02, 0.02]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)探测范围分别为：10,45,30,30
        sizes_list = [10, 45, 30, 30]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)状态空间维度分别为：4,4,5,5
        obs_dim_list = [4, 4, 5, 5]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)动作空间维度分别为：2,2,3,3
        act_dim_list = [2, 2, 3, 3]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)颜色分别如下
        # colors = [(100, 100, 255),(255, 100, 100),(255,150,150),(255,100,100)]
        colors = [(100, 100, 255), (255, 100, 100), (0, 0, 0), (255, 100, 100)]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)诱骗功能除了红B1均为false
        islure = [False, False, True, False]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)干扰功能除了红B2均为false
        isdisturb = [False, False, False, True]

        self.size = sizes_list

        self.agents = [Agent() for i in range(num_agents)]
        for i in range(num_agents):
            # 除了蓝A的速度不可变，其余速度最小值均为0
            self.agents[i].min_speed = speed_list[i] * 0.30864 if i == 0 else 0

            # 最大速度转换为0.1km/min
            self.agents[i].max_speed = speed_list[i] * 0.30864

            # 最大加速度转换为0.1km/min^2
            self.agents[i].accel = accel_list[i] * 36

            # 设定探测范围、状态空间维度、颜色、诱骗功能、干扰功能
            self.agents[i].size = sizes_list[i]
            self.agents[i].obs_dim = obs_dim_list[i]
            self.agents[i].color = colors[i]
            self.agents[i].is_lure = islure[i]
            self.agents[i].is_disturb = isdisturb[i]

            if obs_dim_list[i] == 4:
                self.observation_space.append(Box(low=np.array([-50, -50, self.agents[i].min_speed, -np.pi]),
                                                  high=np.array([50, 50, self.agents[i].max_speed, np.pi]),
                                                  dtype=np.float32))
            elif obs_dim_list[i] == 5:
                if islure[i] == True:
                    self.observation_space.append((Box(low=np.array([-50, -50, self.agents[i].min_speed, -np.pi]),
                                                       high=np.array([50, 50, self.agents[i].max_speed, np.pi]),
                                                       dtype=np.float32),
                                                   Discrete(2)
                                                   ))

                elif isdisturb[i] == True:
                    self.observation_space.append((Box(low=np.array([-50, -50, self.agents[i].min_speed, -np.pi]),
                                                       high=np.array([50, 50, self.agents[i].max_speed, np.pi]),
                                                       dtype=np.float32),
                                                   Discrete(2)
                                                   ))

            if act_dim_list[i] == 2:
                self.action_space.append(Box(low=np.array([0, -np.pi / 2]),
                                             high=np.array([self.agents[i].accel, np.pi / 2]), dtype=np.float32))
            elif act_dim_list[i] == 3:
                if islure[i] == True:
                    self.action_space.append((Box(low=np.array([0, -np.pi / 2]),
                                                  high=np.array([self.agents[i].accel, np.pi / 2]), dtype=np.float32),
                                              Discrete(2)))

                elif isdisturb[i] == True:
                    self.action_space.append((Box(low=np.array([0, -np.pi / 2]),
                                                  high=np.array([self.agents[i].accel, np.pi / 2]), dtype=np.float32),
                                              Discrete(2)))

        self.state = np.zeros(num_agents * 4)
        self.dt = 1 / 60

        self.reset()

        self.window = Window(800, 800, visible=False)
        self.total_distance_blue = 0

    def get_obs(self):
        state = []
        for i in range(1, self.num_agents):
            # 非诱骗、干扰功能智能体，状态包括：坐标x，坐标y，速度，朝向
            if not self.agents[i].is_disturb and not self.agents[i].is_lure:
                state.append(np.array([self.agents[i].state.p_posx, self.agents[i].state.p_posy,
                                       self.agents[i].state.p_vel, self.agents[i].state.p_direct]
                                      ))

            # 诱骗功能智能体，状态包括：坐标x，坐标y，速度，朝向，诱骗
            elif not self.agents[i].is_disturb:
                state.append(np.array([self.agents[i].state.p_posx, self.agents[i].state.p_posy,
                                       self.agents[i].state.p_vel, self.agents[i].state.p_direct,
                                       self.agents[i].state.lure]
                                      ))

            # 干扰功能智能体，状态包括：坐标x，坐标y，速度，朝向，干扰
            elif not self.agents[i].is_lure:
                state.append(np.array([self.agents[i].state.p_posx, self.agents[i].state.p_posy,
                                       self.agents[i].state.p_vel, self.agents[i].state.p_direct,
                                       self.agents[i].state.disturb]
                                      ))
        return state

    def set_actions(self, actions_n):
        for i in range(1, self.num_agents):
            self.agents[i].action.acclr = actions_n[i - 1][0]
            self.agents[i].action.angle_change = actions_n[i - 1][1]

            # 一般来说就一个额外的动作
            if self.agents[i].is_disturb:
                self.agents[i].action.d_disturb = actions_n[i - 1][2]

            if self.agents[i].is_lure:
                self.agents[i].action.d_lure = actions_n[i - 1][2]

            self.exec_actions(i)

    def get_distance(self, agent1, agent2):
        # 获取智能体1坐标
        x1, y1 = agent1.state.p_posx, agent1.state.p_posy

        # 获取智能体2坐标
        x2, y2 = agent2.state.p_posx, agent2.state.p_posy

        # 欧氏距离作为距离的表示
        return np.sqrt(np.sum(np.square(((x1 - x2), (y1 - y2)))))

    # 对指定智能体进行状态调整
    def exec_actions(self, i):
        self.agents[0].lured = False
        for j in range(self.num_agents):
            self.agents[j].size = self.size[j]

        if self.agents[i].state.disturb==1:
            for j in range(self.num_agents):
                if j!=i:
                    # 使用距离相关的缩放因子，来更新智能体 `j` 的探测范围
                    self.agents[j].size=self.size[j]*(1-1/self.get_distance(self.agents[i],self.agents[j]))*0.5

        if self.agents[i].is_lure:
            # 获取诱骗目标的位置
            lure_target_x, lure_target_y = self.agents[i].state.p_posx, self.agents[i].state.p_posy  # 实现此函数以获取诱骗目标的位置

            # 计算朝向诱骗目标的方向角
            target_direction = np.arctan2(lure_target_y - self.agents[0].state.p_posy,
                                          lure_target_x - self.agents[0].state.p_posx)

            # 计算朝向诱骗目标的动作
            # 不太确定朝向到底应不应该加个pi
            desired_angle_change = target_direction - self.agents[0].state.p_direct

            # 让蓝色非智能体直接指向诱骗目标
            self.agents[0].action.angle_change = desired_angle_change + np.pi

            self.agents[0].lured = True

        # 速度变化量=加速度*时间
        self.agents[i].state.p_vel += self.agents[i].action.acclr * self.dt

        # 方向变化量=角速度 * 时间
        self.agents[i].state.p_direct += self.agents[i].action.angle_change * self.dt

        # 位置变化量=速度*sin或者cos*时间
        self.agents[i].state.p_posx += self.agents[i].state.p_vel * np.cos(self.agents[i].state.p_direct) * self.dt
        self.agents[i].state.p_posy += self.agents[i].state.p_vel * np.sin(self.agents[i].state.p_direct) * self.dt

        # 干扰状态 = 动作中干扰的数值 0/1
        self.agents[i].state.disturb = self.agents[i].action.d_disturb

        # 诱骗状态 = 动作中诱骗的数值 0/1
        self.agents[i].state.lure = self.agents[i].action.d_lure

    def set_blue(self):
        # 设定非智能体蓝A的响应状态
        blue = self.agents[0]
        if blue.lured:
            # 诱骗之后向着智能体红B（诱骗功能单元）的方向移动
            self.agents[0].action.acclr, _ = self.action_space[0].sample()
        else:
            self.agents[0].action.acclr, self.agents[0].action.angle_change = self.action_space[0].sample()

        # ======================= 非智能体的基础时间步长移动 ==========================
        # 速度变化量=加速度*时间
        self.agents[0].state.p_vel += self.agents[0].action.acclr * self.dt

        # 方向变化量=角速度 * 时间
        self.agents[0].state.p_direct += self.agents[0].action.angle_change * self.dt

        # 位置变化量=速度*sin或者cos*时间
        self.agents[0].state.p_posx += self.agents[0].state.p_vel * np.cos(self.agents[0].state.p_direct) * self.dt
        self.agents[0].state.p_posy += self.agents[0].state.p_vel * np.sin(self.agents[0].state.p_direct) * self.dt

    def reset(self):
        # Set the initial velocities for the red and blue agents
        initial_velocities = [12.3, 1.8, 1.8, 1.8]

        # Set the initial state of each agent
        for i in range(self.num_agents):
            self.agents[i].state.p_direct = np.random.uniform(-np.pi, np.pi)
            self.agents[i].state.p_vel = initial_velocities[i]

            if i == 1:  # 红A的初始化位置
                self.agents[i].state.p_posx = 0
                self.agents[i].state.p_posy = 0
            elif i == 2:  # 红B1的初始化位置，位于红A左侧
                # redA_x = self.agents[1].state.p_posx
                # redA_y = self.agents[1].state.p_posy
                redB1_x = np.random.uniform(-15, 15)
                redB1_y = np.random.uniform(-15, 15)
                self.agents[i].state.p_posx = redB1_x
                self.agents[i].state.p_posy = redB1_y
            elif i == 3:  # 红B2的初始化位置，位于红A右侧
                # redA_x = self.agents[1].state.p_posx
                # redA_y = self.agents[1].state.p_posy
                redB2_x = -redB1_x
                redB2_y = -redB1_y
                self.agents[i].state.p_posx = redB2_x
                self.agents[i].state.p_posy = redB2_y
            else:
                # 对于其他智能体，随机生成位置
                self.agents[i].state.p_posx = np.random.uniform(-30, 30)
                self.agents[i].state.p_posy = np.random.uniform(-30, 30)

            if self.agents[i].is_lure:
                self.agents[i].state.lure = False

            if self.agents[i].is_disturb:
                self.agents[i].state.disturb = False

        return self.get_obs()

    def step(self, actions_n):
        self.set_actions(actions_n)

        self.set_blue()

        state = self.get_obs()
        reward = self.reward()
        done = [False for _ in range(1, self.num_agents)]

        return np.array(state), np.array(reward), np.array(done)

    def reward(self):
        rewards = []

        # 计算蓝A与所有红B1和红B2之间的距离
        distances_to_red_Bs = [self.get_distance(self.agents[0], self.agents[i]) for i in range(2, self.num_agents)]

        # 找到距离蓝A最近的红B
        closest_red_B_index = np.argmin(distances_to_red_Bs) + 2  # 加2是因为红B从第2个智能体开始

        # 蓝A与最近的红B之间的距离
        distance_to_closest_red_B = distances_to_red_Bs[closest_red_B_index - 2]

        # 定义奖励，当距离越来越近时奖励为正
        reward_for_blue_A = 1.0 / (1.0 + distance_to_closest_red_B)

        # 分配奖励
        for i in range(1, self.num_agents):
            if i == closest_red_B_index:
                rewards.append(reward_for_blue_A)
            else:
                # 其他智能体获得零奖励
                rewards.append(0.0)

        return rewards

    def render(self):
        self.window.set_visible(True)

        @self.window.event
        def on_draw():
            gl.glClearColor(1, 1, 1, 1)
            self.window.clear()

            for i in range(self.num_agents - 1, -1, -1):
                agent = self.agents[i]
                circle = pyglet.shapes.Circle(agent.state.p_posx + 400, agent.state.p_posy + 400, agent.size)
                circle.color = agent.color
                circle.opacity = 128
                circle.draw()

    def sample_random_actions(self):
        actions = []
        for i in range(1, self.num_agents):
            if type(self.action_space[i]) != tuple:
                act = self.action_space[i].sample()
            else:
                act = self.action_space[i][0].sample()

                act = np.append(act, self.action_space[i][1].sample())
            actions.append(act)

        return actions


if __name__ == "__main__":
    env = Environment()
    i = 0
    env.reset()


    def update(dt):
        global i
        if i < 488:
            actions = env.sample_random_actions()
            env.step(actions)
            env.render()
            print(env.get_obs())
            i += 1
        else:
            pyglet.app.exit()


    # Schedule the update function to be called every 0.1 seconds
    pyglet.clock.schedule_interval(update, 0.1)

    # Run the pyglet event loop
    pyglet.app.run()
