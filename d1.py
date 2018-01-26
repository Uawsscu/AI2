#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 23:39:13 2017
@author: sezan92
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 23:39:13 2017
@author: sezan92
"""
# In[1] Libraries
import gym
import numpy as np
import pandas as pd

# In[2]
env = gym.make('MountainCar-v0') #เลือก env
s = env.reset() #initialize
env.render() #ปฎิบัติ

# In[3]
pos_bins = pd.cut([-1.2, 0.6], bins=99, retbins=True)[1] #Car Position ตำแหน่ง min>>>max
vel_bins = pd.cut([-0.07, 0.07], bins=99, retbins=True)[1] #Car Velocity ควาเร็วรถ

allStates= []
for ii in pos_bins:
    for jj in vel_bins:
      allStates.append(np.array([ii, jj]))
allStates = np.array(allStates)
#print(allStates)
def value_to_state(x):
    global allStates
    global pos_bins
    global vel_bins
    xpos = np.digitize(x[0], pos_bins) #บอกว่าอยู่ช่วงไหน
    """
    >>> x = np.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> inds = np.digitize(x, bins)
    >>> inds
    array([1, 4, 3, 2])
    
    >>> for n in range(x.size):
...   print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])
...

    0.0 <= 0.2 < 1.0
    4.0 <= 6.4 < 10.0
    2.5 <= 3.0 < 4.0
    1.0 <= 1.6 < 2.5
    
    """
    xvel = np.digitize(x[1], vel_bins)
    stateValue = np.array([pos_bins[xpos], vel_bins[xvel]])
    state = np.where((allStates == stateValue).all(axis=1))[0][0] #axis 0 ค่าของแนวนอน axis1 คือค่าแนวตั้ง
    return state

# In[4] Q Table
Q = np.zeros([len(allStates),
              env.action_space.n]) #10000x3
lr = 0.8 #เป็นค่าเลอนนิ่งเลทของสูตร ปรับขนาดขนมปังโดยดูจากรางวัล (reword)#เลอนิงเลด
y = 0.95 #ลดขนาดขนมปัง
num_episodes = 30000
rList = []

# In[4]
for i in range(num_episodes):
    s_raw = env.reset()
    s = value_to_state(s_raw)
    rAll =0
    d = False
    j = 0
    while j < 200:
        j += 1
        a = np.argmax(Q[s, :]+np.random.randn(1, env.action_space.n)*(1./(i+1))) #สุ่มเลือก action 0-2 ว่าจะเคลื่อนที่ไปซ้ายหรือขวา
        s1_raw, r, d, _ = env.step(a) #env.step(a)ยืนยันการกระทำ และนำผลของการกระทำไปคำนวณต่อ observation, reward, done, info = env.step(action)
        s1 = value_to_state(s1_raw) #s1_raw คือ start ใหม่ที่ได้จาก a, r, d, _
        """ s1_raw คือ state ใหม่ที่ได้จาก a
            r(reward) เป็น1 เมื่อถึงเส้นชัย
            d (done) สิ้นสุดเมื่อเป็น(gameOver)=true  ไปต่อเมื่อ=false
            """
        env.render() #render and display ออกหน้าจอ
        Q[s, a] = Q[s, a]+lr*(r+y*np.max(Q[s1, :])-Q[s, a])
        #Q[s, a] คือเลือกค่าในช่อง โดยปรับขนาดขนมปังโดยดูจารางวัล
        #Q[s,a]= Q[s,a]+r+y*np.max(Q[s1,:])
        rAll += r
        s = s1


        if d == True or j == 199:
            if d == False:
                print("Not Episode")
                print("Episode " + str(i) + " and Reward " + str(rAll))  #ถ้าเกมจบ แสดง score และจบรอบการเล่นนั้น
            else :
                print("t")

            rList.append(rAll)
            break

            """https://www.youtube.com/watch?v=W3bk2pojLoU"""