import numpy as np
import gym
import tensorflow as tf

# hyperparameters
n_obs = 80 * 80  # dimensionality of observations
h = 200  # hidden layer neurons
n_actions = 3  # number of available actions
learning_rate = 1e-3 #เป็นค่าเลอนนิ่งเลทของสูตร ปรับขนาดขนมปังโดยดูจากรางวัล (reword)#เลอนิงเลด
gamma = .99  # ลดขนาดขนมปัง
decay = 0.99  # decay rate for RMSProp gradients
save_path = 'models/pong.ckpt'

# gamespace
env = gym.make("Pong-v0")  # environment info
s1 = env.reset()
env.render() #ปฎิบัติ

prev_x = None
xList, rList, yList = [], [], []
running_reward = None
rAll = 0
j = 0

# initialize model
tf_model = {}
with tf.variable_scope('layer_one', reuse=False):
    xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(n_obs), dtype=tf.float32)
    tf_model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
with tf.variable_scope('layer_two', reuse=False):
    xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(h), dtype=tf.float32)
    tf_model['W2'] = tf.get_variable("W2", [h, n_actions], initializer=xavier_l2)


# tf operations
def tf_discount_rewards(tf_r):  # tf_r ~ [game_steps,1]
    discount_f = lambda a, v: a * gamma + v;  # รับ gamma=.99 เข้ามาก่อน แล้วเอา a,vทีหลัง
    # print ("ttttt",discount_f,"eee")
    # print("tf.reverse===",tf.reverse(tf_r, [0,0]))
    #print ("r==",tf_r)
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r, [True, False]))  # reverse กลับด้าน
    """
    x = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])

    tf.reverse(x, [0, 1]):
[[[17 18 19]
  [14 15 16]
  [11 12 13]]

 [[ 7  8  9]
  [ 4  5  6]
  [ 1  2  3]]]"""
    tf_discounted_r = tf.reverse(tf_r_reverse, [True, False])
    # print (tf_discounted_r)
    return tf_discounted_r  # Tensor("ReverseV2_1:0", shape=(4,) //4ตัว//, dtype=float32)


# print (tf_discount_rewards([3.,4.,5.,6.]))


def tf_policy_forward(x):  # x ~ [1,D]
    h = tf.matmul(x, tf_model['W1'])
    h = tf.nn.relu(h)
    logp = tf.matmul(h, tf_model['W2'])
    p = tf.nn.softmax(logp)
    return p


# downsampling
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


# tf placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs], name="tf_x")
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions], name="tf_y")
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tf_epr")

# tf reward processing (need tf_discounted_epr for policy gradient wizardry)

tf_discounted_epr = tf_discount_rewards(tf_epr)
tf_mean, tf_variance = tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
tf_discounted_epr -= tf_mean
tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

# tf optimizer op
tf_Softmax = tf_policy_forward(tf_x)
loss = tf.nn.l2_loss(tf_y - tf_Softmax)
#print ("loss",loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
train_op = optimizer.apply_gradients(tf_grads) #เอาเกรเดียนมาปรับค่าเทรน

# tf graph initialization
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# try load saved model
saver = tf.train.Saver(tf.all_variables())
load_was_success = True  # yes, I'm being optimistic
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except:
    print ("no saved model to load. starting new session")
    load_was_success = False
else:
    print ( "loaded model: {}".format(load_path))
    saver = tf.train.Saver(tf.all_variables())
    j = int(load_path.split('-')[-1])

# training loop
while True:
    if True: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(s1)
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs) #รอบแรกมัน null ต้องทำให้มีข้อมูล
    prev_x = cur_x #เปลี่ยนเฟรม

    # stochastically sample a policy from the network
    feed = {tf_x: np.reshape(x, (1, -1))} # none *1 คือ 1 บรรทัด ไม่รู้จบ
    #print (feed)
    """
    np.reshape(3, (1, -1))
    (1 , 2, 3,4)
    """
    #print (tf_aprob)
    aprob = sess.run(tf_Softmax, feed);
    aprob = aprob[0, :] #เอาตำแหน่ง 0-n
    print (aprob)
    if np.random.rand(1) < e:
        a[0] = env.action_space.sample()
    action = np.random.choice(n_actions, p=aprob)
    label = np.zeros_like(aprob);
    label[action] = 1

    # step the environment and get new measurements
    s1, r, d, info = env.step(action + 1)  #env.step(a)ยืนยันการกระทำ และนำผลของการกระทำไปคำนวณต่อ observation, reward, done, info = env.step(action)
    rAll += r  #s1_raw คือ start ใหม่ที่ได้จาก a, r, d, _
    """ s1_raw คือ state ใหม่ที่ได้จาก a
                r(reward) เป็น1 เมื่อได้แต้ม
                d (done) สิ้นสุดเมื่อเป็น(gameOver)=true  ไปต่อเมื่อ=false
                """
    # record game history
    xList.append(x);
    yList.append(label);
    rList.append(r)

    if d:
        # update running reward
        running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01

        # parameter update
        feed = {tf_x: np.vstack(xList), tf_epr: np.vstack(rList), tf_y: np.vstack(yList)}
        #print (feed)  [ 0.,  0.,  0., ...,  0.,  0.,  0.]] //update 80*80 & [ 0.,  1.,  0.], update action

        _ = sess.run(train_op, feed)

        if j % 10 == 0:
            print ('ep {}: reward: {}, mean reward: {:3f}'.format(j, rAll, running_reward))
        else:
            print ('\tep {}: reward: {}'.format(j, rAll))

        # bookkeeping
        xList, rList, yList = [], [], []  # reset game history
        j += 1  # the Next Episode
        s1 = env.reset()  # reset env
        rAll = 0
        if j % 50 == 0: #ถ้ารัน50รอบ
            saver.save(sess, save_path, global_step=j)
            print ("SAVED MODEL #{}".format(j))