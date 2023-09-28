import pyglet
from threading import Thread
from time import sleep
import numpy as np
from pyglet import gl
from gym.spaces import Box,Discrete
from pyglet.window import Window

class Environment:
    def __init__(self, num_agents=4):
        self.num_agents = num_agents
        # Assuming the coordinate values range from -10 to 10,
        # speed values range from -5 to 5, and angle values range from -pi to pi
        self.observation_space = Box(low=np.array([-50, -50, 0, -np.pi]*num_agents, dtype=np.float32),
                                     high=np.array([50, 50, 5, np.pi]*num_agents, dtype=np.float32), dtype=np.float32)

        # 红A智能体的最大速度为0.926km/min，为了省事，保留一位小数
        # 蓝色智能体的最大的速度1.2347km/min
        self.observation_space.high[2] = 9.2

        self.observation_space.low[6] = 12.3
        self.observation_space.high[6] = 12.3

        # 红B智能体的最大速度为0.463-0.5556km/min，这里取0.55
        self.observation_space.high[10] = 5.5
        self.observation_space.high[14] = 5.5

        # Assuming the acceleration values range from -2 to 2, and angle change values range from -pi/4 to pi/4
        self.action_space = Box(low=np.array([0, -np.pi/2]*num_agents), 
                                high=np.array([2, np.pi/2]*num_agents), dtype=np.float32)

        # 蓝色智能体不能加速或者减速
        self.action_space.low[2] = 0
        self.action_space.high[2] = 0

        # 红A的加速度为0.02m/s^2=72m/min^2,红B和红A加速度相同
        self.action_space.high[0] = 0.72
        self.action_space.high[4] = 0.72
        self.action_space.high[6] = 0.72

        # 探测范围，依次是红A，蓝A，红B1，红B2
        self.sizes = [45, 10, 30, 30]

        self.state = np.zeros(num_agents * 4)
        self.dt=1/60

        self.reset()
        

        self.window = Window(800, 800,visible=False)
        # pyglet.clock.schedule_interval(self.update, self.dt)  # Update 120 times per second
                
        # Add a new attribute to track the total distance moved by the blue agent
        self.total_distance_blue = 0


    def reset(self):
        # Set the initial velocities for the red and blue agents
        # 红A的初始速度为11.112km/h=0.1852km/min
        # 蓝A的初始速度为74.08km/h=1.2347km/min
        # 红B的初始速度为11.112km/h=0.1852km/min
        initial_velocities = [1.8, 12.3, 1.8, 1.8]

        # Set the initial state of each agent
        for i in range(self.num_agents):
            x = np.random.uniform(-25, 25)
            y = np.random.uniform(-25, 25)
            angle = np.random.uniform(-np.pi, np.pi)
            v = initial_velocities[i]

            self.state[i*4:i*4+4] = [x, y, v, angle]
        
        return np.array(self.state)

    def step(self, actions):
        
        # Split the actions into acceleration and angle change
        for i in range(self.num_agents):
            x, y, v, angle = self.state[i*4:i*4+4]
            a, angle_change = actions[i]

            # Clip the action to the action space of the agent
            a*=self.action_space.high[i*2]
            angle_change*=self.action_space.high[i*2+1]

            a = np.clip(a, self.action_space.low[i*2], self.action_space.high[i*2])
            angle_change = np.clip(angle_change, self.action_space.low[i*2+1], self.action_space.high[i*2+1])

            # Update the velocity and angle
            v += a * self.dt
            angle += angle_change * self.dt

            # Clip the velocity and angle to the observation space of the agent
            v = np.clip(v, self.observation_space.low[i*4+2], self.observation_space.high[i*4+2])
            angle = np.clip(angle, self.observation_space.low[i*4+3], self.observation_space.high[i*4+3])

            # Update the position
            x += v * np.cos(angle) * self.dt
            y += v * np.sin(angle) * self.dt

            self.state[i*4:i*4+4] = [x, y, v, angle]

        # Add a new attribute to track the total distance moved by the blue agent
        self.total_distance_blue += 12.3*self.dt
        
        reward=self.reward(self.state)
        done=[False for _ in range(self.num_agents)]

        return np.array(self.state),np.array(reward),np.array(done)
    
    # 红色智能体（逃避者）的目标是尽可能远离蓝色智能体（追逐者），而蓝色智能体的目标是尽可能接近红色智能体。
    def reward(self, state):
        # Assuming the state is [x_blue, y_blue, v_blue, angle_blue, x_red, y_red, v_red, angle_red, x_B1, y_B1, v_B1, angle_B1, x_B2, y_B2, v_B2, angle_B2]
        x_blue, y_blue, _, angle_blue, x_red, y_red, _, _, x_B1, y_B1, v_B1, angle_B1, x_B2, y_B2, v_B2, angle_B2 = state

        # Calculate the distance between the blue and red agents
        distance = np.sqrt((x_blue - x_red)**2 + (y_blue - y_red)**2)

        # 首先先计算出来红色智能体与蓝色智能体之间的arctan夹角
        # 然后利用这个角度与实际的角度进行对比，如果实际角度减去智能体的角度，加上π除以2π，获得一个0~360的角度偏差值，再减去个π
        # 作为最终角度上的奖励偏差
        direction_to_red = np.arctan2(y_red - y_blue, x_red - x_blue)
        direction_difference = np.abs((direction_to_red - angle_blue + np.pi) % (2*np.pi) - np.pi)

        # 红色智能体的奖励值应该是与蓝色智能体的距离越大的话，奖励值越大
        reward_red = distance*0.1

        # 蓝色智能体的奖励值应该是与红色智能体的距离越小，奖励值越大
        reward_blue = -distance*0.1 - direction_difference

        # 蓝色智能体的sizes表示的就是其探测的范围，如果在他的探测范围之内的话，其奖励函数应该增加
        if distance <= self.sizes[1]:
            # 如果蓝色智能体能探测到红色智能体的话，那么奖励函数值应该是增加的
            reward_blue += 5

        # 如果二者的距离小于Red_A的范围的话，那么红色智能体的奖励函数值应该会相应的减少
        if distance <= self.sizes[0]:
            # 如果蓝色智能体在红色智能体的探测范围之内的话，那么红色智能体的奖励值是减少的
            reward_red -= 5

        # 红B的reward
        reward_red_B1 = distance*0.1
        reward_red_B2 = distance*0.1

        return reward_red, reward_blue, reward_red_B1, reward_red_B2
    

    def render(self):
        self.window.set_visible(True)
        @self.window.event
        def on_draw():
            gl.glClearColor(1, 1, 1, 1)
            self.window.clear()  # Clear the window with white color
            states = self.state.reshape((self.num_agents, 4))
            coordinates = states[:, :2]
            colors = [(255, 100, 100), (100, 100, 255), (255, 0, 0), (255, 0, 0)]  # Warm red and blue
            
            # Red agent is 4.5 times larger than blue agent
            sizes = self.sizes

            for i, (x, y) in enumerate(coordinates):
                circle = pyglet.shapes.Circle(x+400, y+400, sizes[i])
                circle.color = colors[i]
                circle.opacity = 128  # Set the opacity to make the circle semi-transparent
                circle.draw()
    
    def sample_random_actions(self):
        # Get the shape of the action space
        action_shape = self.action_space.shape[0]

        actions=[]
        for i in range(self.num_agents):
            # Generate a random action for each agent
            actions.append(np.random.uniform(self.action_space.low[i], self.action_space.high[i], action_shape//self.num_agents))

        return actions
if __name__=="__main__":
    env = Environment()
    i = 0
    def update(dt):
        global i
        if i < 488:
            actions = env.sample_random_actions()
            env.step(actions)
            env.render()
            # print("智能体的状态：",env.state[:],"智能体的动作：",actions)
            # print(env.total_distance_blue)
            print(env.state)
            i += 1
        else:
            pyglet.app.exit()

    # Schedule the update function to be called every 0.1 seconds
    pyglet.clock.schedule_interval(update, 0.1)

    # Run the pyglet event loop
    pyglet.app.run()