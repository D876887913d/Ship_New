<!--
 * @Author: gongweijing 876887913@qq.com
 * @Date: 2023-12-05 12:27:42
 * @LastEditors: gongweijing 876887913@qq.com
 * @LastEditTime: 2023-12-05 12:53:14
 * @FilePath: /gongweijing/Ship_New/hyar/readme.md
 * @Description: 
 * 
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
-->
# linux docker里面创建虚拟环境
1. todo

# 复现HyAR说明：
1. 论文的链接为：https://arxiv.org/pdf/2109.05490.pdf
2. 附录材料的链接为：https://openreview.net/forum?id=wQkaGq7Vz6q

# 其余算法的链接：
1. Multi-Pass Deep Q-Networks：https://github.com/cycraig/MP-DQN
2. todo

# 复现训练环境搭建：
```
pip install -r requirements.txt
```

```
pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
pip install -e git+https://github.com/cycraig/gym-goal#egg=gym_goal
pip install -e git+https://github.com/cycraig/gym-soccer#egg=gym_soccer 

```
