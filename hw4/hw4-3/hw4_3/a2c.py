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
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, output_size):
        super(Actor,self).__init__()
        """self.fc1 = nn.Sequential(
            #nn.Linear(84 * 84 * 4, 128),
            #nn.ReLU(),
            #nn.Linear(128, output_size),
            #nn.Sigmoid()
            #nn.Softmax(dim=1)
            nn.Conv2d(4, 32, kernel_size = 8, stride = 4),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
        )"""
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc2 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Sigmoid(),
            nn.Softmax()
        )
        
    def forward(self,observation):
        # observation = observation.view(-1,84,84,4)
        # observation = observation.permute(0, 3, 1, 2)
        #observation = self.fc1(observation)
        print(observation)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        outputs = self.fc2(observation.view(observation.size(0), -1))
        return outputs 

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        """self.fc1 = nn.Sequential(
            #nn.Linear(84 * 84 * 4, 128),
            #nn.ReLU(),
            #nn.Linear(128, output_size),
            #nn.Sigmoid()
            #nn.Softmax(dim=1)
            nn.Conv2d(4, 32, kernel_size = 8, stride = 4),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
        )"""
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc2 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self,observation):
        # observation = observation.view(-1,84,84,4)
        # observation = observation.permute(0, 3, 1, 2)
        #observation = self.fc1(observation)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        outputs = self.fc2(observation.view(observation.size(0), -1))
        return outputs 

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
        self.actor = Actor(self.n_actions).to(device)
        self.critic = Critic().to(device)
        self.target_critic = Critic().to(device)
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(),lr= LR)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(),lr= LR)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.reward_list = []
        self.loss_list = []
        self.test = self.args.test_a2c
        self.critic_loss = nn.MSELoss()

        if args.test_a2c:
            #you can load your model here
            print('loading trained model')
            self.cnn.load_state_dict(torch.load(args.load_model, map_location=device))
            self.actor.load_state_dict(torch.load(args.load_model, map_location=device))
            self.critic.load_state_dict(torch.load(args.load_model, map_location=device))

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
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
       
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            
            #last_screen = self.env.reset()
            #last_screen = torch.tensor(last_screen).to(device)
            current_screen = self.env.reset()
            current_screen = torch.tensor(current_screen).to(device)
            #state = current_screen - last_screen
            state = current_screen
            done = False
            reward_sum = 0
            batch = []
            while not done:
                # Select and perform an action
                action = self.make_action(state)
                #last_screen = current_screen
                current_screen, reward, done, info = self.env.step(action.item())
                current_screen = torch.tensor(current_screen).to(device)
                reward_sum += reward
                reward = torch.tensor([reward]).to(device)

                V_sp1 = self.critic(current_screen)
                V_s = self.critic(state)
                advantage = reward + GAMMA* V_sp1 - V_s
                batch.append((state, action, advantage))

                """if not done:
                    #next_state = current_screen - last_screen
                    next_state = current_screen
                else:
                    next_state = None
                """
                next_state = current_screen

                # Store the transition in memory
                #action = torch.LongTensor(action).to(device).unsqeeze(1)
                self.memory.push(state, action, next_state, reward, torch.FloatTensor([done]).to(device))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                #if self.steps_done % 4 == 0 and len(self.memory) >= 10000:
                self.loss_list.append(self.optimize_model())
                #if self.steps_done % 1000 == 0 and len(self.memory) >= 10000:

            print("Episode: {} | Step: {} | Reward: {}".format(i_episode, self.steps_done, reward_sum), end='\r')
            sys.stdout.write('\033[K')

            self.reward_list.append(reward_sum)

            states, actions, advantages = zip(*batch)


            probs = self.actor(self.cnn(states))
            #print(probs.shape)
            #print(actions.shape)
            #print(rewards.shape)
            actor_loss = nn.BCELoss(weight=advantages)
            loss = actor_loss(probs.squeeze(1),actions)#not sure 
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            if i_episode % 100 == 99:
                print("---------------------------------------------")
                print("Episode:", i_episode)
                print("Latest 30 episode average reward: {:.4f}".format(sum(self.reward_list[-30:])/30))              

            if i_episode%100==99:
                log = {
                  'i_episode' : i_episode,
                  'state_dict' : self.policy_net.state_dict(),
                  'loss' : self.loss_list,
                  'latest_reward' : self.reward_list
                }
                torch.save(log , model_folder_path+ str(i_episode+1)+'.pth.tar')
            


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

