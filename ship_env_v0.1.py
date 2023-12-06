'''
Author: gongweijing 876887913@qq.com
Date: 2023-12-05 19:54:15
LastEditors: gongweijing 876887913@qq.com
LastEditTime: 2023-12-06 01:06:50
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

class Entity:
    """ This is a base entity class, representing moving objects. """
    def __init__(self):
        self.accel_max = 0
        self.speed_max = 0
        self.exlore_size = 0
        
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


class BlueA(Entity):
    def __init__(self):
        super().__init__()
        self.name = 'Non-agent - Blue A'
        self.color = (100, 100, 255)
        self.accel_max = 0
        self.speed_max = segment_to_km_per_second(40)
        self.explore_size = 1
        # 最大航程为10km，转化为km/s之后，最大的可以行驶的时间步为485个。
        self.accel = 0
        self.velocity = segment_to_km_per_second(40)


        
class RedA(Entity):
    def __init__(self):
        super().__init__()
        # 加速度为:0.02m/s^2
        self.name = 'Agent - Red A'
        self.color = (255, 100, 100)
        self.accel_max = 0.02 * 10**(-3)
        self.speed_max = segment_to_km_per_second(30)
        self.explore_size = 4.5


class RedB1(Entity):
    def __init__(self):
        super().__init__()
        self.name = 'Decoy agent - Red B1'
        self.color = (255,0,110)
        # 加速度为:0.02m/s^2
        self.accel_max = 0.02 * 10**(-3)
        self.speed_max = segment_to_km_per_second(15)
        self.explore_size = 2

class RedB2(Entity):
    def __init__(self):
        super().__init__()
        self.name = 'Interfering agent - Red B2'
        self.color = (0,100,100)
        self.accel_max = 0
        self.speed_max = segment_to_km_per_second(17)
        self.explore_size = 1
        self.accel = 0
        self.velocity = segment_to_km_per_second(17)

class ShipEnv(gym.Env):
    def __init__(self):
        self.redA  = None
        self.redB1 = None
        self.redB2 = None
        self.blueA = None
        self.reset()

    def reset(self):
        self.redA  = RedA()
        self.redB1 = RedB1()
        self.redB2 = RedB2()
        self.blueA = BlueA()
        # 假设三个红色智能体初始方向向上走，蓝色智能体在其前方explore_scopeB/3~2*explore_scopeB/3之间的范围
        # 且运动的方向为向下。
        self.redA.position = np.array([0,0])
        self.redA.velocity = segment_to_km_per_second(6)
        self.redA.angle = math.pi/2
        
        self.redB1.position = np.array([segment_to_km_per_second(random.uniform(0.5,1)),0])
        self.redB1.velocity = segment_to_km_per_second(6)
        self.redB1.angle = self.redA.angle # 随便选个角度

        self.redB2.position = np.array([-segment_to_km_per_second(random.uniform(0.8,1.2)),0])
        self.redB2.velocity = segment_to_km_per_second(17)
        self.redB2.angle = self.redA.angle # 随便选个角度

        self.blueA.position = np.array([random.uniform(self.blueA.exlore_size/3,2*self.blueA.explore_size/3),0])
        self.blueA.velocity = segment_to_km_per_second(40)
        self.blueA.angle = - self.redA.angle # 随便选个角度


env = ShipEnv()
env.reset()

window = pyglet.window.Window()

def entity_draw_comment(entity,base_y):
    rA_label = pyglet.text.Label(f'{entity.name}',
                            font_name='Times New Roman',
                            font_size=10,
                            color = (0,0,0,255),
                            x=0, y=window.height-4-base_y,
                            anchor_x='left', anchor_y='top',                          
                            )
    rA_Circle = pyglet.shapes.Circle(250,window.height-12-base_y,7)
    rA_Circle.color = entity.color
    rA_Circle.opacity = 179
    rA_label.draw()
    rA_Circle.draw()

def entity_draw_body(entity):
    draw_pos = [0,0]
    draw_pos[0] = window.width // 2
    draw_pos[1] = window.height // 2     

    rA_Circle = pyglet.shapes.Circle(draw_pos[0],draw_pos[1],entity.explore_size*40)
    rA_Circle.color = entity.color
    rA_Circle.opacity = 179
    rA_Circle.draw()

    rA_Circle = pyglet.shapes.Circle(draw_pos[0],draw_pos[1],1)
    rA_Circle.color = (255,255,255)
    rA_Circle.opacity = 179
    rA_Circle.draw()

@window.event
def on_draw():
    pyglet.gl.glClearColor(1, 1, 1, 1)
    window.clear()
    entity_draw_comment(env.redA,0)
    entity_draw_comment(env.redB1,16)
    entity_draw_comment(env.redB2,16*2)
    entity_draw_comment(env.blueA,16*3)

    entity_draw_body(env.redA)
    entity_draw_body(env.redB1)
    entity_draw_body(env.redB2)
    entity_draw_body(env.blueA)


pyglet.app.run()