'''
Author: gongweijing 876887913@qq.com
Date: 2023-12-05 19:54:15
LastEditors: gongweijing 876887913@qq.com
LastEditTime: 2023-12-08 00:54:44
FilePath: /gongweijing/Ship_New/ship_env_v0.1.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np
import math
import pyglet
import gym
from gym import spaces, error
import random

# 为了方便，这里重构的代码统一将所有的单位转换为:km、s

def segment_to_km_per_second(seg):
    # 1 节 ≈ 1.852 公里/h
    km_h = seg*1.852
    km_s = seg/3600 
    return km_s

def norm(vec2d):
    # from numpy.linalg import norm
    # faster to use custom norm because we know the vectors are always 2D
    assert len(vec2d) == 2
    return math.sqrt(vec2d[0]*vec2d[0] + vec2d[1]*vec2d[1])

def to_deg(rad):
    return rad/(math.pi/180)

class Entity:
    """ This is a base entity class, representing moving objects. """
    def __init__(self):
        self.accel_max = 0
        self.speed_max = 0
        self.explore_size = 0
        
        # 分别对应x,y的值
        self.position =       np.array([0., 0.])
        self.velocity =       0.
        self.accel =          0.
        # 假定角速度固定，船只转向效能有限
        self.angle =          0.
        self.angle_velocity = 0.
        # 像是小船的话最大转向角速度一般都能达到最大九十度
        self.bound_angle_velocity = np.array([-math.pi/2., math.pi/2.])
        # 角度边界限制，防止一直向某个方向变化达到无穷大
        self.bound_angle = np.array([-math.pi, math.pi])
        

    def update(self):
        """ Update the position and velocity. """
        self.position += self.moving()
        self.velocity += self.accel
        self.angle    += self.angle_velocity
        # 角度约束
        self.angle_constraint()
        # 速度约束
        self.velocity_constraint()

    def get_simulate_position(self):
        draw_pos = [0,0]
        draw_pos[0] = window.width // 2
        draw_pos[1] = window.height // 2    
        draw_pos[0] = draw_pos[0] + self.position[0] *DRAW_SCALE
        draw_pos[1] = draw_pos[1] + self.position[1] *DRAW_SCALE
        return draw_pos
        
    
    '''
    description: 获取两个方向速度变化的数值，便于后续计算
    param {*} self
    return {*}
    '''    
    def moving(self):
        delta_x = self.velocity*math.cos(self.angle)
        delta_y = self.velocity*math.sin(self.angle)
        return np.array([delta_x,delta_y])
    
    '''
    description: 设定角度约束，将角度范围控制在-pi~pi之间
    param {*} self
    return {*}
    '''    
    def angle_constraint(self):
        if self.angle < self.bound_angle[0]:
            self.angle += math.pi * 2
        if self.angle > self.bound_angle[1]:
            self.angle -= math.pi * 2

    '''
    description: 设定速度约束，如果超出了给定的速度范围，将其控制在速度范围间
    param {*} self
    return {*}
    '''
    def velocity_constraint(self):
        if self.velocity < 0:
            self.velocity = 0
        if self.velocity > self.speed_max:
            self.velocity = self.speed_max


    def distance(self, other):
        """ Computes the euclidean distance to another entity. """
        return norm(self.position - other.position)
    
    def angle_to_target(self, target):
        delta = target.position - self.position
        delta_mod = norm(delta)
        to_angle = math.acos(delta[0])
        if delta[1] >= 0:
            to_angle = to_angle
        else:
            to_angle = -to_angle
        return to_angle



class BlueA(Entity):
    def __init__(self):
        super().__init__()
        self.name = 'Non-agent - Blue A'
        self.color = (100, 100, 255)
        self.accel_max = 0
        self.speed_max = segment_to_km_per_second(40)
        self.explore_size = 1
        self.confirm_explore_size = 0.1
        self.boom_size = 0.02
        # 最大航程为10km，转化为km/s之后，最大的可以行驶的时间步为485个。
        self.accel = 0
        self.velocity = segment_to_km_per_second(40)
        # 候选红A在candidated_red，find_true_flag=1的时候就不再判断直接去追,black_list是存被标记为红B1的
        self.candidate_red = []
        self.find_true_flag = 0
        self.black_list = []

    def reset(self):        
        self.position = np.array([0,random.uniform(self.explore_size/3,2*self.explore_size/3)])
        self.velocity = segment_to_km_per_second(40)
        self.angle = - math.pi/2 # 随便选个角度
    
    def _update(self):
        for i in self.candidate_red:
            print(f'当前的候选红A的name为：{i.name}')
        if len(self.candidate_red) == 1:
            self.angle_velocity = self.angle_to_target(self.candidate_red[0])-self.angle
        elif len(self.candidate_red) > 1:
            min_dist = None
            min_dist_agent = None
            for i in self.candidate_red:
                if not min_dist_agent:
                    min_dist = self.distance(i)
                    min_dist_agent = i
                if self.distance(i) < min_dist:
                    min_dist = self.distance(i)
                    min_dist_agent = i
            self.angle_velocity = self.angle_to_target(min_dist_agent)-self.angle
            print(f'更近的一个智能体为:{min_dist_agent.name}')
        self.update()
        
    
        
class RedA(Entity):
    def __init__(self):
        super().__init__()
        # 加速度为:0.02m/s^2
        self.name = 'Agent - Red A'
        self.color = (255, 100, 100)
        self.accel_max = 0.02 * 10**(-3)
        self.speed_max = segment_to_km_per_second(30)
        self.explore_size = 4.5

    def reset(self):
        self.position = np.array([0,0])
        self.velocity = segment_to_km_per_second(6)
        self.angle = math.pi/2


class RedB1(Entity):
    def __init__(self):
        super().__init__()
        self.name = 'Decoy agent - Red B1'
        self.color = (255,50,110)
        # 加速度为:0.02m/s^2
        self.accel_max = 0.02 * 10**(-3)
        self.speed_max = segment_to_km_per_second(15)
        self.explore_size = 2

    def reset(self):
        self.position = np.array([random.uniform(0.5,1),0])
        self.velocity = segment_to_km_per_second(6)
        self.angle =  math.pi/2

class RedB2(Entity):
    def __init__(self):
        super().__init__()
        self.name = 'Interfering agent - Red B2'
        self.color = (255,100,50)
        self.accel_max = 0.02 * 10**(-3)
        self.speed_max = segment_to_km_per_second(15)
        self.explore_size = 2

    def reset(self):
        self.position = np.array([-random.uniform(0.8,1.2),0])
        self.velocity = segment_to_km_per_second(6)
        self.angle =  math.pi/2

class ShipEnv(gym.Env):
    def __init__(self):
        self.redA  = None
        self.redB1 = None
        self.redB2 = None
        self.blueA = None
        self.done = False
        self.reset()

    def reset(self):
        self.redA  = RedA()
        self.redB1 = RedB1()
        self.redB2 = RedB2()
        self.blueA = BlueA()
        # 假设三个红色智能体初始方向向上走，蓝色智能体在其前方explore_scopeB/3~2*explore_scopeB/3之间的范围
        # 且运动的方向为向下。
        self.redA.reset()
        self.redB1.reset()
        self.redB2.reset()
        self.blueA.reset()


    def red_in_explore_region(self):
        for i in [self.redA,self.redB1]:
            dist_i = self.blueA.distance(i)                            
            if dist_i < self.blueA.explore_size:
                if i not in self.blueA.candidate_red and i not in self.blueA.black_list:
                    self.blueA.candidate_red.append(i)
            if dist_i < self.blueA.confirm_explore_size:
                if type(i) == RedA:
                    self.blueA.find_true_flag = 1
                else:
                    self.blueA.candidate_list.remove(i)
                    self.blueA.black_list.append(i)
            if dist_i < self.blueA.boom_size:
                self.done = True

    def step(self):
        self.red_in_explore_region()
        self.blueA._update()
            

DRAW_SCALE = 40
DRAW_POINT = 2
env = ShipEnv()
env.reset()

window = pyglet.window.Window()
batch = pyglet.graphics.Batch()


def entity_draw_comment(entity,base_y):
    rA_label = pyglet.text.Label(f'{entity.name}',
                            font_name='Times New Roman',
                            font_size=10,
                            color = (0,0,0,255),
                            x=0, y=window.height-4-base_y,
                            anchor_x='left', anchor_y='top',
                            batch=batch                          
                            )
    rA_Circle = pyglet.shapes.Circle(250,window.height-12-base_y,7,batch=batch)
    rA_Circle.color = entity.color
    rA_Circle.opacity = 128
    batch.draw()

def entity_draw_body():
    draw_group = []    
    draw_group.append(pyglet.shapes.Circle(env.redA.get_simulate_position()[0],env.redA.get_simulate_position()[1],env.redA.explore_size*DRAW_SCALE,batch=batch))
    draw_group[0].opacity = 10
    draw_group[0].color = env.redA.color

    draw_group.append(pyglet.shapes.Circle(env.redB1.get_simulate_position()[0],env.redB1.get_simulate_position()[1],env.redB1.explore_size*DRAW_SCALE,batch=batch))
    draw_group[1].opacity = 128
    draw_group[1].color = env.redB1.color
    
    draw_group.append(pyglet.shapes.Circle(env.redB2.get_simulate_position()[0],env.redB2.get_simulate_position()[1],env.redB2.explore_size*DRAW_SCALE,batch=batch))
    draw_group[2].opacity = 128
    draw_group[2].color = env.redB2.color

    draw_group.append(pyglet.shapes.Circle(env.blueA.get_simulate_position()[0],env.blueA.get_simulate_position()[1],env.blueA.explore_size*DRAW_SCALE,batch=batch))
    draw_group[3].opacity = 128
    draw_group[3].color = env.blueA.color
    batch.draw()

    draw_point = []
    draw_point.append(pyglet.shapes.Circle(env.redA.get_simulate_position()[0],env.redA.get_simulate_position()[1],DRAW_POINT,batch=batch))
    draw_point[0].opacity = 255
    draw_point[0].color = (255,255,255)

    draw_point.append(pyglet.shapes.Circle(env.redB1.get_simulate_position()[0],env.redB1.get_simulate_position()[1],DRAW_POINT,batch=batch))
    draw_point[1].opacity = 255
    draw_point[1].color = (255,255,255)
    
    draw_point.append(pyglet.shapes.Circle(env.redB2.get_simulate_position()[0],env.redB2.get_simulate_position()[1],DRAW_POINT,batch=batch))
    draw_point[2].opacity = 255
    draw_point[2].color = (255,255,255)

    draw_point.append(pyglet.shapes.Circle(env.blueA.get_simulate_position()[0],env.blueA.get_simulate_position()[1],DRAW_POINT,batch=batch))
    draw_point[3].opacity = 255
    draw_point[3].color = (255,255,255)
    batch.draw()


@window.event
def on_draw():
    pyglet.gl.glClearColor(1, 1, 1, 1)
    window.clear()
    entity_list = [env.redA,env.redB1,env.redB2,env.blueA]
    entity_draw_comment(env.redA,0)
    entity_draw_comment(env.redB1,16)
    entity_draw_comment(env.redB2,16*2)
    entity_draw_comment(env.blueA,16*3)

    entity_draw_body()
    # for i in entity_list:
    #     print(i.position)

i = 0
def update(dt):
    global i
    if i < 488:
        env.step()
        if env.done:
            pyglet.app.exit()
        i += 1

pyglet.clock.schedule_interval(update, 0.1)

pyglet.app.run()