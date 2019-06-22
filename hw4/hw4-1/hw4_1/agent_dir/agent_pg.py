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

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold

LR = 0.02
GAMMA = 0.99
EPISODE_n = 40000


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

def DNN(inputs, idx, shape, activate=True):
    W = tf.get_variable("W" + str(idx),shape,tf.float32,tf.random_normal_initializer(mean=0, stddev=0.3))
    B = tf.get_variable("B" + str(idx),shape[-1],tf.float32, tf.constant_initializer(0.1))

    output = tf.matmul(inputs, W) + B
    if activate:
        output = tf.nn.tanh(output)

    return output

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
        environ = environ.unwrapped
        output_graph=True

        #super(Agent_PG,self).__init__(env)
        self.observation_space = prepro(environ.reset())
        self.env = environ
        self.args = args
        self.n_actions = environ.action_space.n
        self.n_features = self.observation_space.shape[0]
        self.lr = LR
        self.gamma = GAMMA

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
        

        self.sess = tf.Session()
        
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = DNN(self.tf_obs, 1,(self.tf_obs.get_shape().as_list()[-1],10))
        # fc2
        all_act = DNN(layer, 2,(10,self.n_actions), False) 
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

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

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        #print(self.all_act_prob.get_shape())
        #print(observation.shape)
        #print(self.tf_obs.get_shape())
        #print(observation[np.newaxis,:].shape)
        #tf_obs = tf.reshape(self.tf_obs,[1,210])
        #observation = observation.reshape(1,210)
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm


    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        #print(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        #print(discounted_ep_rs)
        return discounted_ep_rs


    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # discount and normalize episode reward
        rewards = []
        observation_space = prepro(self.env.reset())

        RENDER = False

        times = 0
        for i_episode in range(EPISODE_n):

            observation = self.env.reset()
            
            while True:
                observation = prepro(observation)
                if RENDER: self.env.render()
                #print(observation.shape)
                action = self.make_action(observation)
                  
                observation_, reward, done, info = self.env.step(action)
                #print (reward)
                
                self.store_transition(observation, action, reward)
                
                if done:
                    ep_rs_sum = sum(self.ep_rs)

                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                     
                    rewards.append(running_reward) 

                    if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
                    print("episode:", i_episode, "  reward:", int(running_reward))

                    vt = self.learn()
                    
                    if i_episode%10 == 9:
                        with open('pong_reward_maxclip/reward_log_'+str(EPISODE_n)+'e.txt','a') as f:
                            f.write("episode:{:4d}, average reward:{:+.5f}\n".format(i_episode+1, sum(rewards[-30:])/30))
                        save_path = "pong_model_maxclip/"+str(i_episode+1)
                        self.saver.save(self.sess,save_path)

                    break

                observation = observation_


    
