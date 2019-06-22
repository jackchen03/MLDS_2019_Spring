import gym
import h5py
import tensorflow as tf
import numpy as np 
from RL_brain_PP_noreduct import PolicyGradient
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
EPISODE_n = 40000

rewards = []

env = gym.make('Pong-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

def prepro(observation):
   observation = observation[35:195]
   observation = observation[::2, ::2, 0]
   observation[observation==144] = 0
   observation[observation != 0] = 1
   
   return observation.astype(np.float).ravel()


observation_space = prepro(env.reset())

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    output_graph=True,
)
times = 0
for i_episode in range(EPISODE_n):

    observation = env.reset()
    
    while True:
        observation = prepro(observation)
        if RENDER: env.render()
        #print(observation.shape)
        action = RL.choose_action(observation)
          
        observation_, reward, done, info = env.step(action)
        #print (reward)
        
        RL.store_transition(observation, action, reward)
        
        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
             
            rewards.append(running_reward) 

            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()
            
            if i_episode%10 == 9:
                with open('RL_brain_reward/reward_log_noreward'+str(EPISODE_n)+'e.txt','a') as f:
                    f.write("episode:{:4d}, average reward:{:+.5f}\n".format(i_episode+1, sum(rewards[-30:])/30))
                '''if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()'''
                '''init = tf.global_variables_initializer()
                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(init)
                    save_path = saver.save(sess, "RL_brain_model/episode"+str(i_episode+2)+".ckpt")'''
            break

        observation = observation_
  
        #save model
        '''if i_episode%10==9:
            #RL.save('RL_brain_model/episode'+str(i_episode+1)+".h9")
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init)
                save_path = saver.save(sess, "RL_brain_model/episode"+str(i_episode+2)+".ckpt")'''
