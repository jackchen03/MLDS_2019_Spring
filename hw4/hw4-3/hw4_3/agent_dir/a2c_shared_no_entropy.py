import os
import sys
import gym
import math
import random
from random import choices
import numpy as np
from agent_dir.agent import Agent
from PIL import Image
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5
LR = 1.5e-4
NUM_EPISODES = 50000

os.environ['CUDA_VISIBLE_DEVIES'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        #Saves a transition.
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Actor_Critic(nn.Module):
    def __init__(self, output_size):
        super(Actor_Critic,self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.actor_fc = nn.Linear(512, output_size)
        self.critic_fc = nn.Linear(512, 1)
        
    def forward(self,observation):
        observation = observation.view(-1,84,84,4)
        observation = observation.permute(0, 3, 1, 2)
        #observation = self.fc1(observation)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = F.relu(self.fc(observation.view(observation.size(0), -1)))

        action_dist = self.actor_fc(observation)
        action_dist = Categorical(F.softmax(action_dist,dim=1))

        value = self.critic_fc(observation.view(observation.size(0), -1))
        
        return action_dist, value


class Agent_A2C(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_A2C,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.args = args
        self.n_actions = self.env.action_space.n
        #self.actor = Actor(self.n_actions).to(device)
        #self.critic = Critic().to(device)
        self.A2C = Actor_Critic(self.n_actions).to(device)
        if args.load_model:
            #you can load your model here
            print('loading trained model')
            self.checkpoint = torch.load(args.load_model, map_location=device)
            self.A2C.load_state_dict(self.checkpoint['state_dict'])
            self.reward_list = self.checkpoint['latest_reward']
            self.loss_list = self.checkpoint['loss']
            
        self.optimizer = optim.RMSprop(self.A2C.parameters(),lr= LR)
        #self.actor_optimizer = optim.RMSprop(self.actor.parameters(),lr= LR)
        #self.critic_optimizer = optim.RMSprop(self.critic.parameters(),lr= LR)
        #self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.reward_list = []
        self.loss_list = []
        self.test = self.args.test_a2c
        #self.critic_loss = nn.MSELoss()

        if args.test_a2c:
            #you can load your model here
            print('loading trained model')
            self.checkpoint = torch.load(args.load_model, map_location=device)
            print(len(self.checkpoint['latest_reward']))
            print(sum(self.checkpoint['latest_reward'][-30:])/30)
            self.A2C.load_state_dict(self.checkpoint['state_dict'])
            
            #self.critic.load_state_dict(torch.load(args.load_model, map_location=device))

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.env.seed(12345)


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        # Get number of actions from gym action space      
        model_folder_path = "models/"+self.args.save_model+"/"
        os.makedirs(model_folder_path,exist_ok=True)

        num_episodes = NUM_EPISODES
       
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            
            state = self.env.reset()
            done = False
            reward_sum = 0
            log_probs = []
            values = []
            rewards = []
            masks = []
            probs = []
            while not done:
                # Select and perform an action
                state = torch.FloatTensor(state).to(device)
                dist, value = self.A2C(state)

                action = dist.sample()
                next_state , reward, done, info = self.env.step(action.item())
                reward_sum += reward
                prob = dist.probs
                log_prob = dist.log_prob(action).unsqueeze(0)

                log_probs.append(log_prob)
                probs.append(prob)
                values.append(value)
                rewards.append(torch.FloatTensor([reward]).to(device))
                masks.append(torch.FloatTensor([1-done]).to(device))

                # Move to the next state
                state = next_state

            print("Episode: {} | Step: {} | Reward: {}".format(i_episode, self.steps_done, reward_sum), end='\r')
            #sys.stdout.write('\033[K')

            self.reward_list.append(reward_sum)

            next_state = torch.FloatTensor(next_state).to(device)
            _ , next_value = self.A2C(next_state)
            returns = self.compute_returns(next_value, rewards, masks)

            probs = torch.cat(probs)
            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns)
            values = torch.cat(values)
            rewards = torch.cat(rewards)

            #next_values = torch.cat([values[1:], next_value], dim = 0)
            #Q = rewards + next_values

            advantage = returns - values

            dist_entropy = -(log_probs * probs).sum(-1).mean()

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            self.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            #actor_loss.backward(retain_graph=True)
            self.optimizer.step()

            #self.optimizer.zero_grad()
            #critic_loss.backward()
            #self.optimizer.step()

            #self.actor_optimizer.zero_grad()
            #self.critic_optimizer.zero_grad()
            #actor_loss.backward()
            #critic_loss.backward()
            #self.actor_optimizer.step()
            #self.critic_optimizer.step()
            if self.args.load_model:
                episode_num = i_episode + self.checkpoint['i_episode']
            else:
                episode_num = i_episode

            if i_episode % 100 == 99:
                print("---------------------------------------------")
                print("Episode:", episode_num)
                print("Latest 30 episode average reward: {:.4f}".format(sum(self.reward_list[-30:])/30))              

            if i_episode%100==99:
                log = {
                  'i_episode' :episode_num,
                  'state_dict' : self.A2C.state_dict(),
                  'loss' : self.loss_list,
                  'latest_reward' : self.reward_list
                }
                torch.save(log , model_folder_path+ str(episode_num+1)+'.pth.tar')
            
    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def make_action(self, observation):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        distribution = self.actor(observation)
        choices(self.env.action_space, distribution)         

        return choices   


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        """# Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.uint8).to(device)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        """
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        current_q = self.critic(state_batch).gather(1,action_batch)
        next_q = reward_batch + (1 - done_batch) * GAMMA * self.target_critic(next_state_batch).detach().max(-1)[0]
        next_q = next_q.unsqueeze(-1)
        #print(reward_batch.shape)
        #print(current_q.shape)
        #print(next_q.shape)
        loss = self.critic_loss(current_q,next_q)

        """# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        #print("POLICY: {}".format(state_action_values))
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE).to(device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        #print("EXPECTED: {}".format(expected_state_action_values))
        # Compute Huber loss
        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = self.loss_func(state_action_values, expected_state_action_values.unsqueeze(1))
        """

        # Optimize the model
        self.critic_optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()
        return loss.item()

