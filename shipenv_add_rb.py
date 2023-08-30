import pyglet
from threading import Thread
from time import sleep
import numpy as np
from pyglet import gl
from gym.spaces import Box,Discrete
from pyglet.window import Window

class Environment:
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        # Assuming the coordinate values range from -10 to 10,
        # speed values range from -5 to 5, and angle values range from -pi to pi
        self.observation_space = Box(low=np.array([-50, -50, 0, -np.pi]*num_agents), 
                                     high=np.array([50, 50, 5, np.pi]*num_agents), dtype=np.float32)

        # 红A智能体的最大速度为0.926km/min，为了省事，保留一位小数
        # 蓝色智能体的最大的速度1.2347km/min    
        self.observation_space.low[2] = 12.3
        self.observation_space.high[2] = 12.3

        self.observation_space.high[6] = 9.2

        # Assuming the acceleration values range from -2 to 2, and angle change values range from -pi/4 to pi/4
        self.action_space = Box(low=np.array([0, -np.pi/2]*num_agents), 
                                high=np.array([2, np.pi/2]*num_agents), dtype=np.float32)

        # 蓝色智能体不能加速或者减速
        self.action_space.low[0] = 0
        self.action_space.high[0] = 0

        self.action_space.high[2] = 0.72

        self.sizes = [10,45] 

        self.state = np.zeros(num_agents * 4)       
        self.dt=1/60

        self.reset()
        

        self.window = Window(800, 800,visible=False)
        # pyglet.clock.schedule_interval(self.update, self.dt)  # Update 120 times per second
                
        # Add a new attribute to track the total distance moved by the blue agent
        self.total_distance_blue = 0


    def reset(self):
        # Set the initial velocities for the red and blue agents
        initial_velocities = [12.3,1.8]

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
            if i==0:
                x, y, v, angle = self.state[i*4:i*4+4]

                # TODO 这个地方写上你的蓝色非智能体策略
                a, angle_change = np.random.uniform(self.action_space.low[i], self.action_space.high[i],2)

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

            else:
                i-=1
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
                i+=1

        # Add a new attribute to track the total distance moved by the blue agent
        self.total_distance_blue += 12.3*self.dt
        
        reward=self.reward(self.state)
        done=[False for _ in range(self.num_agents)]

        return np.array(self.state),np.array(reward),np.array(done)
    
    # 红色智能体（逃避者）的目标是尽可能远离蓝色智能体（追逐者），而蓝色智能体的目标是尽可能接近红色智能体。
    def reward(self, state):
        # Assuming the state is [x_blue, y_blue, v_blue, angle_blue, x_red, y_red, v_red, angle_red]
        x_blue, y_blue, _, angle_blue, x_red, y_red, _, _ = state

        # Calculate the distance between the blue and red agents
        distance = np.sqrt((x_blue - x_red)**2 + (y_blue - y_red)**2)

        # 首先先计算出来红色智能体与蓝色智能体之间的arctan夹角
        # 然后利用这个角度与实际的角度进行对比，如果实际角度减去智能体的角度，加上π除以2π，获得一个0~360的角度偏差值，再减去个π
        # 作为最终角度上的奖励偏差
        direction_to_red = np.arctan2(y_red - y_blue, x_red - x_blue)
        direction_difference = np.abs((direction_to_red - angle_blue + np.pi) % (2*np.pi) - np.pi)

        # 红色智能体的奖励值应该是与蓝色智能体的距离越大的话，奖励值越大
        reward_red = distance*0.1

        # 如果二者的距离小于Red_A的范围的话，那么红色智能体的奖励函数值应该会相应的减少
        if distance <= self.sizes[0]:
            # 如果蓝色智能体在红色智能体的探测范围之内的话，那么红色智能体的奖励值是减少的
            reward_red -= 5

        return reward_red
    

    def render(self):
        self.window.set_visible(True)
        @self.window.event
        def on_draw():
            gl.glClearColor(1, 1, 1, 1)
            self.window.clear()  # Clear the window with white color
            states = self.state.reshape((self.num_agents, 4))
            coordinates = states[:, :2]
            colors = [(100, 100, 255),(255, 100, 100)]  # Warm red and blue
            
            # Red agent is 4.5 times larger than blue agent
            sizes=self.sizes

            for i, (x, y) in enumerate(coordinates):
                circle = pyglet.shapes.Circle(x+400, y+400, sizes[i])
                circle.color = colors[i]
                circle.opacity = 128  # Set the opacity to make the circle semi-transparent
                circle.draw()
    
    def sample_random_actions(self):
        # Get the shape of the action space
        action_shape = self.action_space.shape[0]

        actions=[]
        for i in range(1,self.num_agents):
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
            i += 1
        else:
            pyglet.app.exit()

    # Schedule the update function to be called every 0.1 seconds
    pyglet.clock.schedule_interval(update, 0.1)

    # Run the pyglet event loop
    pyglet.app.run()