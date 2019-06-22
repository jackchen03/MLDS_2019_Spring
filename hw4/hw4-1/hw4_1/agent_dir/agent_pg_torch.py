from agent_dir.agent import Agent
import os
import scipy
import scipy.misc
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    o = o[::2,::2,0]
    o[o == 144] = 0
    o[o == 109] = 0
    o[o != 0] = 1
    
    return o.astype(np.float).ravel()

class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(80 * 80, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            #nn.Softmax()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)

    def forward(self,observation):
        #observation = observation.view(-1,80*80)
        observation = self.fc(observation)
        return observation #p for up

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.args = args
        self.model = PolicyModel().to(device)
        self.optim = optim.Adam(self.model.parameters(),lr= self.args.lr, betas = (0.5, 0.99))
        self.last_state = None #for make_actions ONLY
        self.reward_record = []

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model.load_state_dict(torch.load(args.load_model, map_location=device))

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        np.random.seed(12345)
        torch.manual_seed(12345)
        self.env.seed(12345)

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        model_folder_path = "models/"+self.args.model_name+"/"
        os.makedirs(model_folder_path,exist_ok=True)

        try:
            os.remove("loss_log.txt")
        except OSError:
            pass

        for i in range(self.args.episodes):
            print("Starting episode {}".format(i+1))
            last_state = self.env.reset()
            action = self.env.get_random_action()
            state, _, _, _ = self.env.step(action)

            n_rounds = 0
            done = False
            episode_reward_sum = 0
            batch = []

            while not done:
                delta_state = prepro(state) - prepro(last_state)
                last_state = state

                action = self.make_action(delta_state,test=False)
                state, reward, done, info = self.env.step(action)
                batch.append((delta_state, action_map[action], reward))
                
                episode_reward_sum += reward
                if reward != 0:
                    n_rounds += 1
                    #imageio.imwrite('game.png',delta_state)

            print("Episode {} finished after {} rounds".format(i, n_rounds))
            self.reward_record.append(episode_reward_sum)
            print("Total reward: {:.0f}".format(episode_reward_sum))
            print("Latest 30 episodes average reward: {}".format(sum(self.reward_record[-30:])/30))
            print('---------------------------------------------------------------')

            states, actions, rewards = zip(*batch)
            rewards = self.discount(rewards)
            rewards -= np.mean(rewards)
            rewards /= np.std(rewards)

            self.model.train()

            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)

            probs = self.model(states)
            #print(probs.shape)
            #print(actions.shape)
            #print(rewards.shape)
            loss_function = nn.BCELoss(weight=rewards)
            loss = loss_function(probs.squeeze(1),actions)#not sure 
            loss.backward()
            self.optim.step()
            if i%10==9:
                with open("loss_log.txt","a") as f:
                    f.write("episode: {:4d}, loss: {:+.5f}, avg_r(-30): {:+.5f}\n".format(i+1,loss.item(),sum(self.reward_record[-30:])/30))
            if i%100==99:
                torch.save(self.model.state_dict(), model_folder_path+ str(i+1)+'.pkl')

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
        self.model.eval()
        if test:
            state = prepro(observation)
            if self.last_state is None:
                delta_state = state
            else:
                delta_state = state - self.last_state
            self.last_state = state
        else:
            delta_state = observation

        delta_state = torch.FloatTensor(delta_state).to(device)
        self.model.eval()
        p = self.model(delta_state)[0]
        #p_dist = torch.Tensor([1-p, p]) #[p for UP, p for DOWN]
        action = UP_ACTION if np.random.uniform() < p else DOWN_ACTION

        return action 

    def discount(self, rewards):
        gamma = self.args.gamma
        discounted_rewards = np.zeros_like(rewards)
        run_add = 0
        for t,_ in enumerate(rewards[::-1]):
            if rewards[t] != 0:
                run_add = 0
            run_add = gamma*run_add + rewards[t]
            discounted_rewards[t] = run_add
        return discounted_rewards
