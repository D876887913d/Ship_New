<!--
 * @Author: gongweijing 876887913@qq.com
 * @Date: 2023-12-04 13:33:29
 * @LastEditors: gongweijing 876887913@qq.com
 * @LastEditTime: 2023-12-12 19:06:49
 * @FilePath: /gongweijing/Ship_New/readme.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# 获取训练结果
tensorboard --logdir=./logs


# maddpg的基本依赖方法
## env.action_space
[Discrete(3), Discrete(5)]
## env.observation_space
[Box(-inf, inf, (3,), float32), Box(-inf, inf, (11,), float32)]
## list(obs.values()), list(reward.values()), list(done.values()), list(info.values())
([array([0.15, 0.65, 0.15], dtype=float32), array([ 0.        ,  0.        ,  0.2631022 ,  0.35862896, -0.06338122,-0.11295114,  0.35266367, -0.16499558,  1.        ,  0.        ,0.        ]dtype=float32)]

[-0.01677513865695685, -0.01677513865695685], 

[False, False], 

[{}, {}]
)

# Linux下进行项目部署
```
pip install -r requirements.txt
sudo apt-get install libglu1-mesa-dev
python3 shipenv_add_rb.py
```
# 计划的的一些任务
1. 利用老师给的资料进行hybrid action space reinforce learning算法的测试;
2. 对现有的奖励函数进行多样化加权，增加训练算法的可行性;
3. 添加蓝A抓住红A的情况的结束条件;
4. 增加更容易让各个智能体得到训练的初始化条件。

# 当前想做的工作
1. 跑通[hyar](https://zhuanlan.zhihu.com/p/596495689)相关的算法，找出一个可行的算法备选进行动作空间的搜索，并确保算法本身无问题。
2. 训练一些其中附带的算法，例如p_ddpg等算法，保证算法多样性。
3. 理解其中环境的设定，熟悉内部的环境部署算法。

## HyAR项目学习
1. 熟悉项目基本内容,了解项目基本框架。
2. 熟悉环境配置，掌握其环境的基本搭建方法。
3. 熟悉HyAR算法的基本框架，基本部署到现有环境。

## 查阅相关文献确定船只追捕之类的奖励函数设定经验
1. todo

## 了解多智能体调优方案
1. todo

## 待改动bug
1. 红b的速度跟加速度;

## 算法配置
### 纯规则算法
1. 未发现蓝A的时候采用固定的巡航路线进行移动;
2. 

## 环境设定
### 数制转换
1. 0.1km对应的是环境中的数值1;
2. 采用的数值对应的时间单位为/min；
3. 

### 简化部分
1. 红A初始化为地图中心,即初始的坐标值设定为(0,0)，随机选取速度方向，初始速度值根据给定的 √
1. 红B(诱骗)假设为一直开启诱骗，只要进入蓝A的范围之内，那么就会将其认定为红A，进行追逐。 
2. 红b(干扰)假设为一直开启干扰，变为一个0~1的数值，设定最大的干扰范围为:m，干扰的实际距离则为：0~m(m=0.8km),
进入干扰范围后的探测范围公式为:e_dist = e_dist * (1 - 1/d);
3. 蓝A初始化的范围为: 距离其圆心的距离为:[e_{dist}_b/3,2*e_{dist}_b/3]
4. 红B与红A初始距离为:0.5~1km距离的随机位置;红b与红A初始距离为:0.8~1.2km距离的随机位置;
二者的具体位置的话一个相对于运动方向的左方,一个在相对于红A在右边;


### 添加部分
1. 给每个智能体的中心添加一个标识，例如三角、矩形等，加上注释。
2. 智能体的观测空间应当再包含进去其可以观测到的智能体的状态.

## 未来目标
1. 蓝A仍非智能体，后续修改为2个红b和红B两种情况；
2. 蓝A为智能体应当学习到要去追的目标；
3. 


## 一些能想到要改的点：

### 绘图部分
1. 由于红A的范围基本能囊括所有，所以设定为空心圆，圆心处绘制红色小点；
2. 绘图坐标尽可能与画圆的统一，回头检查一下具体设定有无问题；
3. 

### 智能体彼此间可观测部分配置
1. 蓝A:红A的坐标，红B1坐标  （如果这两个不在观测范围内的话，设定为np.array([-1.,-1.]）
2. 红A:蓝A的坐标            （如果不在观测范围内的话，      设定为np.array([-1.,-1.]）
3. 红B1:蓝A的坐标，红A的坐标（如果这两个不在观测范围内的话，设定为np.array([-1.,-1.]）
4. 红B2:无观测能力，如何判断蓝A进入了干扰范围？


从编程的角度而言，各个智能体的坐标都是知道的，所以进行探测范围access的过程中，应当放到何处就比较重要了。 移动之前可以先判断一下，对于蓝色智能体的纯规则的可以先写一个相应的自动脚本，这个的话一般好理解。

### 蓝色智能体
1. 计算角度的变化量应该是多少，假设候选的A的坐标为A,那么B指向A的向量则为$\vec{BA}=\vec{OA}-\vec{OB}$。最终要达到的角度应当为acos(BA),当前的角度为angle_B,要想从angle_B变为acos(BA),那么就要：

$$angle_B = angle_B + acos(BA)-angle_B$$

## 奖励
### 红色A智能体
红A与蓝A的距离越远，那么红A越容易取得胜利，所以奖励函数与距离的数值成正比，可以直接设定为distance。
### 红色B1智能体
红B1的作用是进行诱骗，所以除了与红A之间的距离成正比之外，按理来说还需要其更接近蓝A，所以可以采用类似于加权的形式进行相应的求解:
$$reward = distance(RedA,BlueA) - 0.5 * distance(RedB1,BlueA)$$
### 红色B2智能体
红B2的作用是进行干扰，所以除了与红A之间的距离成正比之外，按理来说还需要其更大程度的减少蓝A的探测范围，所以可以采用类似于加权的形式进行相应的求解:
$$reward=distance(RedA,BlueA) + 0.5*(BlueA.explore\_size - BlueA.init\_explore\_size)$$

## 老师的思路
先获取一个候选的redA_access的list或者是dict，对应的数据类型应当为entity，要添加的方法还包括获取与其他entity的距离。

先确定下初始队形，把初始化函数写了：
中心是红A
红B1, B2在A两侧，对称，给定一个距离（你们定）
红b1, b2也在A两侧，对称，也给定一个距离（你们定）


蓝A基本设定：
	蓝A永远追1）距离他最近的且2）目前最能被确认是红A的目标（即已经在100m内确认是真红A，他就不会再改变追踪目标）
	蓝A在初始化的时候，不要投放到红b的干扰范围内（设置一个干扰无效截断距离），且探测范围内至少有一个红A或红B

代码：
find_true_flag = 0
初始化一个ID(推测ID)，状态，时间的观测值缓存库dict，存储各个目标被最后一次看到的观测值dict[候选红A1],dict[候选红A2],dict[候选红A3],dict[真实红A],dict[红B1],dict[红B2],dict[红b1],dict[红b2]，包括其位置，航行方向和速度信息，如已经不在蓝A的观测范围内，则按照最后一次见到其的状态（假设方向不改变，匀速），推断其位置

每个step
计算当前的探测范围（可能被红b干扰）
更新dict[候选红A1],dict[候选红A2],,dict[候选红A3],dict[真实红A],dict[红B1_1],dict[红B2_1],dict[红b1],dict[红b2]（被确定非真实红A*的移动到红B1_1，原红A删除；两个相邻时间步目标可计算出是否是同一个ID，只更新值不用更新key；一段时间步没有出现又突然出现的目标，无法推断是否是同一个ID，可以新建一个key:value

while(蓝A行程还有余量）：
if dict[真实红A]存在：
观测得到或推断当前真实红A的方位（认为其方向不改变）存到目标方位
if 距离小于20m:
  蓝A引爆
当前时间步，向目标方位全速（最大加速度）移动


if dict[真实红A]不存在，dict[候选红A*]存在：
观测或推断找到当前距离最近的候选红A*，存到目标方位 #按这个逻辑，默认就会追距离最近的候选红A(需要切换就切换）
if 距离小于100m:
判断是否是真实红A，更新dict(更改身份，删减标签），find_true_flag = 1
if 距离小于20m && find_true_flag == 1：
    蓝A引爆
当前时间步，向目标方位全速（最大加速度）移动

if dict[真实红A]不存在，dict[候选红A*]不存在或已经排除嫌疑，dict[红B2b_1*]存在：
得到红B2_1的初始状态，假设这个转态即为红A的初始状态，假设红A（-90或90转向），得到其目标的方位：
找到当前距离最近的红b*，存到目标方位
当前时间步，向目标方位全速（最大加速度）移动
