'''
Author: gongweijing 876887913@qq.com
Date: 2023-12-08 22:50:41
LastEditors: gongweijing 876887913@qq.com
LastEditTime: 2023-12-09 11:34:22
FilePath: /Ship_New/env_mpe_req.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from maddpg.Env import MAenv
import numpy as np

env = MAenv('simple_speaker_listener', False)
print(env.reset())
act = np.array([[0.65, 0.15, 0.15],[0.65, 0.15, 0.15,0.65, 0.15]])
print(env.action_space)
print(env.observation_space)
print(env.step(act))
