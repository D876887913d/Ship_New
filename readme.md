<!--
 * @Author: gongweijing 876887913@qq.com
 * @Date: 2023-12-04 13:33:29
 * @LastEditors: gongweijing 876887913@qq.com
 * @LastEditTime: 2023-12-04 23:42:15
 * @FilePath: /root/Ship_New/readme.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Linux下进行项目部署
```
pip install -r requirements.txt
sudo apt-get install libglu1-mesa-dev
python3 shipenv_add_rb.py
```
# 之后要进行的一些任务
1. 利用老师给的资料进行hybrid action space reinforce learning算法的测试;
2. 对现有的奖励函数进行多样化加权，增加训练算法的可行性;
3. 添加蓝A抓住红A的情况的结束条件;
4. 增加更容易让各个智能体得到训练的初始化条件。

# 当前想做的工作
1. 跑通[hyar](https://zhuanlan.zhihu.com/p/596495689)相关的算法，找出一个可行的算法备选进行动作空间的搜索，并确保算法本身无问题。
2. 训练一些其中附带的算法，例如p_ddpg等算法，保证算法多样性。
3. 理解其中环境的设定，熟悉内部的环境部署算法。