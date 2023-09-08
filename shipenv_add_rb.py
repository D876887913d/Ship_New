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
        self.p_direct=None

        self.lure=None
        self.disturb=None

# action of the agent
class Action(object):
    def __init__(self):
        self.acclr = None
        self.angle_change = None

        self.d_disturb=None
        self.d_lure=None

class Agent():
    # maxspeed单位是节  accel单位是m/s^2
    def __init__(self):
        super(Agent, self).__init__()
        self.movable = True
        self.max_speed = None
        self.accel =None
        self.size=None
        self.color=None

        self.obs_dim=None

        self.is_lure=False
        self.is_disturb=False
       
        self.state = AgentState()
        self.action = Action()

        self.action_callback = None

class Environment:

    def __init__(self, num_agents=4):
        self.num_agents = num_agents

        self.observation_space=[]
        self.action_space=[]
        self.action_space=[]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)
        # 蓝A，红A，红B1(诱骗)，红B2(干扰)最大速度分别为：40节、30节、18节、18节
        speed_list = [40,30,  18,  18]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)最大加速度分别为：0,0.02,0.02,0.02
        accel_list = [0, 0.02,0.02,0.02]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)探测范围分别为：10,45,30,30
        sizes_list = [10,45,  30,  30] 

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)状态空间维度分别为：4,4,5,5
        obs_dim_list=[4, 4,   5,   5]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)动作空间维度分别为：2,2,3,3
        act_dim_list=[2, 2,   3,   3]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)颜色分别如下
        colors = [(100, 100, 255),(255, 100, 100),(255,150,150),(255,100,100)]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)诱骗功能除了红B1均为false
        islure=[False,False,True,False]

        # 蓝A，红A，红B1(诱骗)，红B2(干扰)干扰功能除了红B2均为false
        isdisturb=[False,False,False,True]

        self.size=sizes_list
            

        self.agents=[Agent() for i in range(num_agents)]
        for i in range(num_agents):
            # 除了蓝A的速度不可变，其余速度最小值均为0
            self.agents[i].min_speed=speed_list[i]*0.30864 if i==0 else 0

            # 最大速度转换为0.1km/min
            self.agents[i].max_speed=speed_list[i]*0.30864

            # 最大加速度转换为0.1km/min^2
            self.agents[i].accel=accel_list[i]*36

            # 设定探测范围、状态空间维度、颜色、诱骗功能、干扰功能
            self.agents[i].size=sizes_list[i]
            self.agents[i].obs_dim=obs_dim_list[i]
            self.agents[i].color=colors[i]
            self.agents[i].is_lure=islure[i]
            self.agents[i].is_disturb=isdisturb[i]

            if obs_dim_list[i]==4:
                self.observation_space.append(Box(low=np.array([-50, -50,self.agents[i].min_speed,-np.pi]), 
                                                  high=np.array([50, 50, self.agents[i].max_speed, np.pi]), dtype=np.float32))
            elif obs_dim_list[i]==5:
                if islure[i]==True:
                    self.observation_space.append((Box(low=np.array([-50, -50,self.agents[i].min_speed,-np.pi]), 
                                                  high=np.array([50, 50, self.agents[i].max_speed, np.pi]), dtype=np.float32),
                                                  Discrete(2)
                                                  ))
                    
                elif isdisturb[i]==True:
                    self.observation_space.append((Box(low=np.array([-50, -50,self.agents[i].min_speed,-np.pi]), 
                                                  high=np.array([50, 50, self.agents[i].max_speed, np.pi]), dtype=np.float32),
                                                  Discrete(2)
                                                  ))

            if act_dim_list[i]==2:
                self.action_space.append(Box(low=np.array([0, -np.pi/2]), 
                          high=np.array([self.agents[i].accel, np.pi/2]), dtype=np.float32))
            elif act_dim_list[i]==3:
                if islure[i]==True:
                    self.action_space.append((Box(low=np.array([0, -np.pi/2]), 
                          high=np.array([self.agents[i].accel, np.pi/2]), dtype=np.float32),Discrete(2)))
                    
                elif isdisturb[i]==True:
                    self.action_space.append((Box(low=np.array([0, -np.pi/2]), 
                          high=np.array([self.agents[i].accel, np.pi/2]), dtype=np.float32),Discrete(2)))

        self.state = np.zeros(num_agents * 4)       
        self.dt=1/60

        self.reset()
        
        self.window = Window(800, 800,visible=False)
        self.total_distance_blue = 0

    def get_obs(self):
        state=[]
        for i in range(1,self.num_agents):
            # 非诱骗、干扰功能智能体，状态包括：坐标x，坐标y，速度，朝向
            if not self.agents[i].is_disturb and not self.agents[i].is_lure:
                state.append(np.array([self.agents[i].state.p_posx, self.agents[i].state.p_posy,
                                      self.agents[i].state.p_vel,self.agents[i].state.p_direct]
                                      ))
                
            # 诱骗功能智能体，状态包括：坐标x，坐标y，速度，朝向，诱骗
            elif not self.agents[i].is_disturb:
                state.append(np.array([self.agents[i].state.p_posx, self.agents[i].state.p_posy,
                                      self.agents[i].state.p_vel,self.agents[i].state.p_direct,
                                      self.agents[i].state.lure]
                                      ))
                
            # 干扰功能智能体，状态包括：坐标x，坐标y，速度，朝向，干扰
            elif not self.agents[i].is_lure:
                state.append(np.array([self.agents[i].state.p_posx, self.agents[i].state.p_posy,
                                      self.agents[i].state.p_vel,self.agents[i].state.p_direct,
                                      self.agents[i].state.disturb]
                                      ))
        return state
    
    def set_actions(self,actions_n):
        for i in range(self.num_agents):
            if i==0:
                self.agents[i].action.acclr,self.agents[i].action.angle_change=self.action_space[0].sample()
            else:
                self.agents[i].action.acclr=actions_n[i-1][0]
                self.agents[i].action.angle_change=actions_n[i-1][1]

                # 一般来说就一个额外的动作
                if self.agents[i].is_disturb:
                    self.agents[i].action.d_disturb=actions_n[i-1][2]
                
                if self.agents[i].is_lure:
                    self.agents[i].action.d_lure=actions_n[i-1][2]

            self.exec_actions(i)

    def get_distance(self,agent1,agent2):
        # 获取智能体1坐标
        x1,y1=agent1.state.p_posx,agent1.state.p_posy

        # 获取智能体2坐标
        x2,y2=agent2.state.p_posx,agent2.state.p_posy

        # 欧氏距离作为距离的表示
        return np.sqrt(np.sum(np.square(((x1-x2),(y1-y2)))))

    # 对指定智能体进行状态调整
    def exec_actions(self,i):
        # 速度变化量=加速度*时间
        self.agents[i].state.p_vel+=self.agents[i].action.acclr*self.dt

        # 方向变化量=角速度 * 时间
        self.agents[i].state.p_direct+=self.agents[i].action.angle_change*self.dt

        # 位置变化量=速度*sin或者cos*时间
        self.agents[i].state.p_posx+=self.agents[i].state.p_vel*np.cos(self.agents[i].state.p_direct) * self.dt
        self.agents[i].state.p_posy+=self.agents[i].state.p_vel*np.sin(self.agents[i].state.p_direct) * self.dt

        # 干扰状态 = 动作中干扰的数值 0/1
        self.agents[i].state.disturb=self.agents[i].action.d_disturb

        # 诱骗状态 = 动作中诱骗的数值 0/1
        self.agents[i].state.lure=self.agents[i].action.d_lure
        
        if self.agents[i].state.disturb==1:
            for j in range(self.num_agents):
                if j!=i:
                    # 使用距离相关的缩放因子，来更新智能体 `j` 的探测范围
                    self.agents[j].size=self.size[j]*(1-1/self.get_distance(self.agents[i],self.agents[j]))*0.5

        if self.agents[i].state.lure==1:
            # 获取诱骗目标的位置
            lure_target_x, lure_target_y = self.agents[i].state.p_posx,self.agents[i].state.p_posy  # 实现此函数以获取诱骗目标的位置

            # 计算朝向诱骗目标的方向角
            target_direction = np.arctan2(lure_target_y - self.agents[0].state.p_posy, lure_target_x - self.agents[0].state.p_posx)

            # 计算朝向诱骗目标的动作
            desired_angle_change = target_direction - self.agents[0].state.p_direct

            # 让蓝色非智能体直接指向诱骗目标
            self.agents[0].action.angle_change = desired_angle_change

    
    def reset(self):
        # Set the initial velocities for the red and blue agents
        initial_velocities = [12.3,1.8,1.8,1.8]

        # Set the initial state of each agent
        for i in range(self.num_agents):
            self.agents[i].state.p_posx = np.random.uniform(-25, 25)
            self.agents[i].state.p_posy = np.random.uniform(-25, 25)
            self.agents[i].state.p_direct = np.random.uniform(-np.pi, np.pi)
            self.agents[i].state.p_vel = initial_velocities[i]
            if self.agents[i].is_lure:
                self.agents[i].state.lure=False
            
            if self.agents[i].is_disturb:
                self.agents[i].state.disturb=False
        
        return self.get_obs()

    def step(self, actions_n):
        self.set_actions(actions_n)

        state=self.get_obs()
        reward=self.reward()
        done=[False for _ in range(1,self.num_agents)]

        return np.array(state),np.array(reward),np.array(done)
    

    def reward(self):
        return [0 for i in range(1,self.num_agents)]
    

    def render(self):
        self.window.set_visible(True)
        @self.window.event
        def on_draw():
            gl.glClearColor(1, 1, 1, 1)
            self.window.clear() 

            for i in range(self.num_agents-1,-1,-1):
                agent=self.agents[i]
                circle=pyglet.shapes.Circle(agent.state.p_posx+400,agent.state.p_posy+400,agent.size)
                circle.color=agent.color
                circle.opacity=128
                circle.draw()
    
    def sample_random_actions(self):
        actions=[]
        for i in range(1,self.num_agents):
            if type(self.action_space[i])!=tuple:
                act=self.action_space[i].sample()
            else:
                act=self.action_space[i][0].sample()
            
                act=np.append(act,self.action_space[i][1].sample())
            actions.append(act)

        return actions
    
if __name__=="__main__":
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