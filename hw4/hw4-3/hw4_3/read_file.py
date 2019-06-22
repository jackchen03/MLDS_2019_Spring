import numpy as np
import matplotlib.pyplot as plt
import torch

#filename1 = 'double_duel_dqn_50000.pth.tar'
#filename2 = 'duel_dqn_50000.pth.tar'
#filename3 = 'double_dqn_50000.pth.tar'
#filename4 = 'cnn_relu_50000.pth.tar'
filename = 'models/a2c_shared_2/95299.pth.tar'


#checkpoint1 = torch.load(filename1,map_location=torch.device('cpu'))
#checkpoint2 = torch.load(filename2,map_location=torch.device('cpu'))
#checkpoint3 = torch.load(filename3,map_location=torch.device('cpu'))
#checkpoint4 = torch.load(filename4,map_location=torch.device('cpu'))
checkpoint1 = torch.load(filename,map_location=torch.device('cpu'))

x_1 = np.linspace(0,45000,45000)


x1 = checkpoint1['i_episode']
y1_reward = checkpoint1['latest_reward']
y1_reward_new = []
max_double_duel = max(y1_reward)

#x2 = checkpoint2['i_episode']
#y2_reward = checkpoint2['latest_reward']
#y2_reward_new = []
#max_duel = max(y2_reward)

#x3 = checkpoint3['i_episode']
#y3_reward = checkpoint3['latest_reward']
#y3_reward_new = []
#max_double = max(y3_reward)

#x4 = checkpoint4['i_episode']
#y4_reward = checkpoint4['latest_reward']
#y4_reward_new = []
#max_cnn_relu = max(y4_reward)

for idx in range(45000):
	y1_reward_new.append(sum(y1_reward[idx-30:idx])/30)
#   y2_reward_new.append(sum(y2_reward[idx-30:idx])/30)
#	y3_reward_new.append(sum(y3_reward[idx-30:idx])/30)
#	y4_reward_new.append(sum(y4_reward[idx-30:idx])/30)

# y1_reward_new.append(sum(y1_reward[-30:])/30)
# y2_reward_new.append(sum(y2_reward[-30:])/30)
# y3_reward_new.append(sum(y3_reward[-30:])/30)
# y4_reward_new.append(sum(y4_reward[-30:])/30)




plt.figure()
plt.plot(x_1, y1_reward_new)
plt.show()
#plt.plot(x_1, y2_reward_new)
#plt.show()
#plt.plot(x_1, y3_reward_new)
#plt.show()
#plt.plot(x_1, y4_reward_new)
#plt.show()
# x = []
# y = []
# with open(filename) as file:
#    for line in file:
#       line1 = line.strip('episode:').split(',')
#       x_tmp = line1[0].strip()
#       line2 = line1[1].strip(' average reward:')
#       y_tmp = line2.strip()
#       x.append(int(x_tmp))
#       y.append(float(y_tmp))

# x_arr = np.array(x[-735:])
# y_arr = np.array(y[-735:])
# print(x_arr)
# print(y_arr)
# file.close()

# plt.figure()
# plt.plot(x_arr,y_arr)
# plt.xlabel('episode')
# plt.ylabel('average rewards')
# plt.show()

