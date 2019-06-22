import os
import sys
import gym
import math
import random
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
class QNet(nn.Module):
    def __init__(self, output_size):
        super(QNet,self).__init__()
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
            nn.Linear(512, output_size)
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal(m.weight)

    def forward(self,observation):
        observation = observation.view(-1,84,84,4)
        observation = observation.permute(0, 3, 1, 2)
        #observation = self.fc1(observation)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        outputs = self.fc2(observation.view(observation.size(0), -1))
        return outputs 


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.args = args
        self.n_actions = self.env.action_space.n
        self.policy_net = QNet(self.n_actions).to(device)  # Q
        self.target_net = QNet(self.n_actions).to(device)  # Q_head
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr= LR, betas = (0.5, 0.99))
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr= 0.00015)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.reward_list = []
        self.test = self.args.test_dqn
        self.loss_func = nn.MSELoss()

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
        if args.load_model:
            self.policy_net.load_state_dict(torch.load(args.load_model, map_location=device))

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        #self.env.seed(12345)


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
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        num_episodes = NUM_EPISODES
       
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
            while not done:
                # Select and perform an action
                action = self.make_action(state, self.test)
                #last_screen = current_screen
                current_screen, reward, done, info = self.env.step(action.item())
                current_screen = torch.tensor(current_screen).to(device)
                reward_sum += reward
                reward = torch.tensor([reward]).to(device)

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
                if self.steps_done % 4 == 0:
                    self.optimize_model()
                #if self.steps_done % 1000 == 0 and len(self.memory) >= 10000:
                if self.steps_done % 1000 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            print("Episode: {} | Step: {} | Reward: {}".format(i_episode, self.steps_done, reward_sum), end='\r')
            sys.stdout.write('\033[K')

            self.reward_list.append(reward_sum)

            if i_episode % 100 == 99:
                print("---------------------------------------------")
                print("Episode:", i_episode)
                print("Latest 30 episode average reward: {:.4f}".format(sum(self.reward_list[-30:])/30))              

            if i_episode%100==99:
                torch.save(self.policy_net.state_dict(), model_folder_path+ str(i_episode+1)+'.pkl')
            


    def make_action(self, observation, test):
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
        if test:
            #eturn self.mapping[self.policy_net(observation).argmax(1).item()]
            #return self.policy_net(observation).argmax(1).item()+1
            return self.policy_net(observation).max(1)[1].view(1, 1).item()
        else:
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.steps_done / EPS_DECAY)
            #if (len(self.memory) >= 10000): self.steps_done += 1
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.policy_net(observation).max(1)[1].view(1, 1)
                    #return self.policy_net(observation).argmax(1).item()
                    # if test:
                    #     return self.policy_net(observation).max(1)[1].view(1, 1).item()
                    # else:
                    #     return self.policy_net(observation).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]]).to(device)
                #return self.mapping[torch.tensor([random.randrange(0,2)]).to(device)]
                #print(self.mapping[random.randrange(0,2)])
                #return self.mapping[random.randrange(0,2)]
                #return random.randrange(0,2)
                # if test:
                #     return self.env.get_random_action()
                # else:
                #     return torch.tensor([[random.randrange(self.n_actions)]]).to(device)
            


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
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

        current_q = self.policy_net(state_batch).gather(1,action_batch)
        next_q = reward_batch + (1 - done_batch) * GAMMA * self.target_net(next_state_batch).detach().max(-1)[0]
        next_q = next_q.unsqueeze(-1)
        #print(reward_batch.shape)
        #print(current_q.shape)
        #print(next_q.shape)
        loss = self.loss_func(current_q,next_q)

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
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
       












