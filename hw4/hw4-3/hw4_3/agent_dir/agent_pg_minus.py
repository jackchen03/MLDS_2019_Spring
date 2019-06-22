from agent_dir.agent import Agent
import os
import scipy
import scipy.misc
import imageio
import numpy as np
import tensorflow as tf
import gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold

LR_A = 0.001
LR_C = 0.01
GAMMA = 0.9
EPISODE_n = 40000
RENDER = False
MAX_EP_STEPS = 1000


UP_ACTION = 2
DOWN_ACTION = 3
action_map = {
    DOWN_ACTION:0,
    UP_ACTION:1
}

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    
    '''y = 0.2126*o[:,:,0] + 0.7152*o[:,:,1] + 0.0722*o[:,:,2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)'''

    o = o[35:195]
    o = o[::2, ::2, 0]
    o[o==144] = 0
    o[o != 0] = 1
   
    return o.astype(np.float).ravel()

def DNN_1(inputs, idx, shape, activate=True):
    W = tf.get_variable("W" + str(idx),shape,tf.float32,tf.random_normal_initializer(mean=0., stddev=0.1))
    B = tf.get_variable("B" + str(idx),shape[-1],tf.float32, tf.constant_initializer(0.1))

    output = tf.matmul(inputs, W) + B
    if activate:
        output = tf.nn.relu(output)

    return output

def DNN_2(inputs, idx, shape, activate=True):
    W = tf.get_variable("W" + str(idx),shape,tf.float32,tf.random_normal_initializer(mean=0., stddev=0.1))
    B = tf.get_variable("B" + str(idx),shape[-1],tf.float32, tf.constant_initializer(0.1))

    output = tf.matmul(inputs, W) + B
    if activate:
        output = tf.nn.softmax(output)

    return output


class Actor(object):
    def __init__ (self, sess, n_features, n_actions, lr=0.0001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            l1 = DNN_1(self.s, 1, (self.s.get_shape().as_list()[-1],20), True)
            self.acts_prob = DNN_2(l1, 2, (20, n_actions), True)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0,self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1,n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1,1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = DNN_1(self.s, 3, (self.s.get_shape().as_list()[-1],20), True)
            self.v = DNN_2(l1, 4, (20,1), True)

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})

        return td_error

class Agent_PG(Agent):       
    def __init__(
            self,
            env, 
            args,
 
    ):  
        """
        Initialize every things you need here.
        For example: building your model
        """
        environ = gym.make('Pong-v0')
        #environ = environ.seed(1)
        environ = environ.unwrapped
        output_graph = False

        #super(Agent_PG,self).__init__(env)
        self.observation_space = prepro(environ.reset())
        self.env = environ
        self.args = args
        self.n_actions = environ.action_space.n
        self.n_features = self.observation_space.shape[0]
        self.lr_a = LR_A
        self.lr_c = LR_C
        self.gamma = GAMMA
        self.render = RENDER

        #self.track_r = []

        self.sess = tf.Session()
        self.actor = Actor(self.sess, n_features=self.observation_space.shape[0], n_actions=self.n_actions, lr=self.lr_a)
        self.critic = Critic(self.sess, n_features=self.observation_space.shape[0], lr=self.lr_c)
        
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    # def _build_net(self):
    #     with tf.name_scope('inputs'):
    #         self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
    #         self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
    #         self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
    #     # fc1
    #     layer = DNN(self.tf_obs, 1,(self.tf_obs.get_shape().as_list()[-1],10))
    #     # fc2
    #     all_act = DNN(layer, 2,(10,self.n_actions), False) 
    #     self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

    #     with tf.name_scope('loss'):
    #         # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
    #         #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
    #         # or in this way:
    #         neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
    #         loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

    #     with tf.name_scope('train'):
    #         self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        #np.random.seed(12345)
        #torch.manual_seed(12345)
        self.env.seed(12345)


    def store_transition(self, r):
        self.track_r.append(r)

    def train(self):

        rewards = []

        self.init_game_setting()

        for i_episode in range(EPISODE_n):
            s = self.env.reset()
            t = 0
            track_r = []

            while True: 
                s_tmp_1 = prepro(s) #s_tmp_1 = (6400,) s = (210,160,3) 

                if self.render : self.env.render()

                a = self.actor.choose_action(s_tmp_1)

                s_, r, done, info = self.env.step(a) #s_ = (210,160,3)

                if(t == 0) :
                    s_tmp_2 = prepro(s_) #s_tmp = (6400,)
                else:
                    s_tmp_2 = prepro(s_- s)

                #if done: r = -21
    
                track_r.append(r)


                td_error = self.critic.learn(s_tmp_1,r,s_tmp_2)
                self.actor.learn(s_tmp_1,a,td_error)

                s = s_ - s
                t += 1

                #if done or t>= MAX_EP_STEPS:
                if done:
                    #print(done)
                    ep_rs_sum = sum(track_r)

                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                     
                    rewards.append(running_reward) 

                    if running_reward > DISPLAY_REWARD_THRESHOLD: self.render = True     # rendering
                    print("episode:", i_episode, "  reward:", int(running_reward))
                    
                    if i_episode%10 == 9:
                        with open('pong_reward_minus/reward_log_'+str(EPISODE_n)+'e.txt','a') as f:
                            f.write("episode:{:4d}, average reward:{:+.5f}\n".format(i_episode+1, sum(rewards[-30:])/30))
                        save_path = "pong_model_minus/"+str(i_episode+1)
                        self.saver.save(self.sess,save_path)

                    break



    
    
