'''
Author: gongweijing 876887913@qq.com
Date: 2023-12-05 19:54:15
LastEditors: gongweijing 876887913@qq.com
LastEditTime: 2023-12-05 22:34:19
FilePath: /gongweijing/Ship_New/ship_env_v0.1.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np
import math
import pyglet
import gym
from gym import spaces, error

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
        self.name = '非智能体-蓝A'
        self.accel_max = 0
        self.speed_max = segment_to_km_per_second(40)
        self.exlore_size = 1
        # 最大航程为10km，转化为km/s之后，最大的可以行驶的时间步为485个。
        self.accel = 0
        self.velocity = segment_to_km_per_second(40)

        
class RedA(Entity):
    def __init__(self):
        super().__init__()
        # 加速度为:0.02m/s^2
        self.name = '智能体-红A'
        self.accel_max = 0.02 * 10**(-3)
        self.speed_max = segment_to_km_per_second(30)
        self.exlore_size = 4.5


class RedB1(Entity):
    def __init__(self):
        super().__init__()
        self.name = '诱骗智能体-红B1'
        # 加速度为:0.02m/s^2
        self.accel_max = 0.02 * 10**(-3)
        self.speed_max = segment_to_km_per_second(15)
        self.exlore_size = 2

class RedB2(Entity):
    def __init__(self):
        super().__init__()
        self.name = '干扰智能体-红B2'
        self.accel_max = 0
        self.speed_max = segment_to_km_per_second(17)
        self.exlore_size = 1
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

env = ShipEnv()
env.reset()

window = pyglet.window.Window()
# pyglet.font.add_file('STFANGSO.TTF')
# font_name = u'华文仿宋'.encode('gbk')
label = pyglet.text.Label(f'{env.redA.name}',
                          font_name=font_name,
                          font_size=12,
                          x=0, y=window.height,
                          anchor_x='left', anchor_y='top')
@window.event
def on_draw():
    window.clear()
    label.draw()

pyglet.app.run()