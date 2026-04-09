import torch
from scipy.special import jv
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from torch.distributions import Categorical, Dirichlet, Beta
import torch.nn.functional as F
from geopy.distance import geodesic
import h3
import numpy as np
from numba import njit
import math
import gc
from collections import deque
import random
# 记录开始时间

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")
class BasicBuffer:
    def __init__(self,max_size):
        self.max_size=max_size
        self.buffer=deque(maxlen=max_size)
    def push(self,state,action,reward,next_state,done):
        experience=(state,action,np.array([reward]),next_state,done)
        self.buffer.append(experience)
    def sample(self,batch_size):
        state_batch=[]
        action_batch=[]
        reward_batch=[]
        next_state_batch=[]
        done_batch=[]

        batch=random.sample(self.buffer,batch_size)
        for experience in batch:
            state,action,reward,next_state,done=experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        return (state_batch,action_batch,reward_batch,next_state_batch,done_batch)
    def __len__(self):
        return len(self.buffer)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.beam_index = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.action_dim = action_dim

    def forward(self, state):
        """无噪声 logits，用于训练（Critic 需要稳定 Q）"""
        enc = self.net(state)
        logits = self.beam_index(enc)
        logits = torch.tanh(logits) * 10   # 限制在 [-2,2]
        actions, aa = gumbel_topk(logits,8)#gumbel_top_k_sampling(logits)
        return actions, aa, logits

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.q1_net = nn.Sequential(
             nn.Linear(state_dim + 40,512),
             nn.ReLU(),
             nn.Linear(512,128),
             nn.ReLU(),
             nn.Linear(128,1),
             )
        self.q2_net = nn.Sequential(
             nn.Linear(state_dim + 40,512),
             nn.ReLU(),
             nn.Linear(512,128),
             nn.ReLU(),
             nn.Linear(128,1),
             )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim = -1)
        return self.q1_net(sa), self.q2_net(sa)
def sample_gumbel(shape, eps=1e-20, device="cpu"):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_topk(logits, k, tau=1.0, hard=False):
    device = logits.device
    g = sample_gumbel(logits.shape, device=device)
    z = logits + g
    y_soft = torch.softmax(z / tau, dim=-1)
    _, topk_indices = z.topk(8, dim=-1)

    if not hard:
        return z,topk_indices
    
    # hard top-k
    _, topk = y_soft.topk(k, dim=-1)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, topk, 1.0)
    
    # straight-through
    return y_hard + (y_soft - y_soft.detach()),topk_indices

def compute_entropy_for_known_actions_vectorized(logits, selected_actions):
    """
    向量化计算已知动作的熵（无 for 循环）

    参数:
        logits: [B, action_dim] 未归一化的动作 logits
        selected_actions: [B, topk] 已知的 topk 动作索引

    返回:
        mean_entropy: [B] 每个样本的 topk 动作平均熵
    """
    B, action_dim = logits.shape
    topk = selected_actions.size(1)
    device = logits.device
    selected_actions = selected_actions.long()
    # 初始化掩码（全部为 False）
    mask = torch.zeros((B, action_dim), dtype=torch.bool, device=device)

    # 扩展 logits 以匹配 topk 维度 [B, topk, action_dim]
    expanded_logits = logits.unsqueeze(1).expand(-1, topk, -1)

    # 生成所有步骤的掩码 [B, topk, action_dim]
    # 使用 scatter_ 的 cumsum 技巧一次性生成所有掩码
    mask_scatter = torch.zeros((B, topk, action_dim), dtype=torch.bool, device=device)
    mask_scatter.scatter_(2, selected_actions.unsqueeze(-1), True)
    mask_cumulative = mask_scatter.cumsum(dim=1) > 0  # 累积掩码

    # 应用掩码（填充 -inf）
    masked_logits = expanded_logits.masked_fill(mask_cumulative, -1e6)

    # 计算 softmax 和熵 [B, topk]
    probs = F.softmax(masked_logits, dim=-1)
    entropies = Categorical(probs).entropy()  # [B, topk]

    # 返回平均熵 [B]
    return entropies.mean(dim=1)
class SACAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, alpha=0.2):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.target_critic = Critic(state_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.replay_buffer=BasicBuffer(max_size=100000)
        self.epsilon=1
        self.epsilon_increment=0.999/10000
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=3e-5)  # 3e-4   1e-4 3e-5(clip0.95 3e-5 0.99)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=3e-5)  # 5e-4 5e-4  1e-4
        self.gamma = 0.95
        self.tau = 5e-3
        self.batch_size = 64
        self.learn_step_counter = 0
        self.device = device
        self.gumbel_tau =1.0

        self.target_entropy = torch.log(torch.tensor(8))
        self.log_alpha = torch.tensor(np.log(alpha),requires_grad=True)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=3e-5)

    def learn(self):
        self.learn_step_counter+=1
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        states      = torch.FloatTensor(states).to(self.device)
        #print(actions)
        #actions     = torch.FloatTensor(actions).to(self.device)   # DDPG 动作一般是连续，注意 Float
        actions     = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), 1)


        # ---- 2. 计算 target Q 值 ----

        q1, q2 = self.critic(states, actions)
        alpha = self.log_alpha.exp()
        
        with torch.no_grad():
            next_actions, next_aas, next_logits = self.actor(next_states)
            q1_target, q2_target = self.target_critic(next_states, next_actions)
            q_target_min = torch.min(q1_target, q2_target)
            entropy = compute_entropy_for_known_actions_vectorized(next_logits, next_aas).unsqueeze(1)  
            
            target_q = rewards + self.gamma * (1 - dones) * (q_target_min - alpha * entropy)
        
        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)
        critic_loss = loss_q1 + loss_q2
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # ---- 4. 更新 actor ----
        new_actions, new_aas, logits = self.actor(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        entropy = compute_entropy_for_known_actions_vectorized(logits, new_aas).unsqueeze(1)  
       
        # actor loss
        actor_loss = (alpha * entropy - q_pi).mean()

        # alpha loss (自动温度)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.optim_actor.zero_grad()
        self.optim_alpha.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        alpha_loss.backward()
        self.optim_alpha.step()

            # ---- 5. 软更新 target 网络 ----
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        '''
        if self.learn_step_counter % 500 == 0:
            torch.save(self.actor.state_dict(), 'model_actor_SAC_3.5.pth')
            torch.save(self.critic.state_dict(), 'model_critic_SAC_3.5.pth')    
            torch.save(self.target_critic.state_dict(), 'model_target_critic_SAC_3.5.pth') 
            torch.save(self.optim_alpha.state_dict(), 'model_optim_alpha_SAC_3.5.pth') 
            print(critic_loss.item())
            print(self.learn_step_counter)
        '''


def channel_gain(satellite_location, user_location, user_num, cell_num, T_s):
    theta_3dB = 1.25 * np.pi / 180
    Gr_max = 20
    Gs_max = 36.2
    PL_o = 7
    f = 20 * 10 ** 3  # MHz
    [sat_lat, sat_lon] = satellite_location[T_s] / 180 * np.pi
    # 转相对坐标系
    lon_u = user_location[:, 0] / 180 * np.pi
    lat_u = user_location[:, 1] / 180 * np.pi
    cell_lon = np.array(cell_location[:, 1]) / 180 * np.pi
    cell_lat = np.array(cell_location[:, 0]) / 180 * np.pi
    '''
    d_su = np.zeros(user_num)
    PL = np.zeros(user_num)
    d_sc = np.zerps(cell_num)
    d_cu = np.zeros([cell_num,user_num])
    theta = np.zeros([cell_num,user_num])
    r = np.zeros([cell_num,user_num])
    G_s = np.zeros([cell_num,user_num])
    H = np.zeros([cell_num,user_num])
    '''
    x_sat = (6371 + 508) * np.cos(sat_lat) * np.cos(sat_lon)
    y_sat = (6371 + 508) * np.cos(sat_lat) * np.sin(sat_lon)
    z_sat = (6371 + 508) * np.sin(sat_lat)
    x_cell = 6371 * np.cos(cell_lat) * np.cos(cell_lon)
    y_cell = 6371 * np.cos(cell_lat) * np.sin(cell_lon)
    z_cell = 6371 * np.sin(cell_lat)

    x_user = 6371 * np.cos(lat_u) * np.cos(lon_u)
    y_user = 6371 * np.cos(lat_u) * np.sin(lon_u)
    z_user = 6371 * np.sin(lat_u)
    d_sc = np.sqrt((x_sat[None, None] - x_cell[:, None]) ** 2 +
                   (y_sat[None, None] - y_cell[:, None]) ** 2 +
                   (z_sat[None, None] - z_cell[:, None]) ** 2).squeeze()

    # 计算 d_su（卫星到用户的距离）
    d_su = np.sqrt((x_sat[None, None] - x_user) ** 2 +
                   (y_sat[None, None] - y_user) ** 2 +
                   (z_sat[None, None] - z_user) ** 2)

    # 计算路径损耗 PL
    PL = -20 * np.log10(d_su) - 20 * np.log10(f) - 20 * np.log10(4 * np.pi * 10 ** 9 / (3 * 10 ** 8)) - PL_o

    # 计算 d_cu（基站到用户的距离）
    d_cu = np.sqrt((x_cell[:, None] - x_user) ** 2 +
                   (y_cell[:, None] - y_user) ** 2 +
                   (z_cell[:, None] - z_user) ** 2)

    # 计算 theta
    theta = np.arccos((d_su ** 2 + d_sc[:, None] ** 2 - d_cu ** 2) / (2 * d_su * d_sc[:, None]))

    # 计算 r
    r = 2.07123 * np.sin(theta) / np.sin(theta_3dB)

    # 计算 G_s
    G_s = np.where(r == 0, Gs_max,
                   10 * np.log10(10 ** (Gs_max / 10) * (jv(1, r) / (2 * r) + 36 * (jv(3, r) / r ** 3)) ** 2))
    # 计算 H
    H = 10 ** ((Gr_max + G_s + PL) / 10)
    
    return H


def channel_gain_T(satellite_location, user_location, user_num, cell_num):
    T_s = 30
    H = []
    for t in range(T_s):
        H.append(channel_gain(satellite_location, user_location, user_num, cell_num, t))
    return H

def update_queue_single(queues, transfer_size):
    queues = np.array(queues, dtype=np.float64)
    num_packets = queues.shape[0]
    # new_queue = queues.copy()
    new_queue = np.empty_like(queues)  # 创建同样形状的数组（未初始化）
    new_queue[:] = queues
    first_non_zero = -1

    if transfer_size <= 0:
        first_non_zero = 0 if queues.any() else -1
        return new_queue, first_non_zero

    cumsum = 0
    for j in range(num_packets):
        if cumsum >= transfer_size:
            break

        remaining = transfer_size - cumsum
        transmit = min(remaining, queues[j])
        new_queue[j] -= transmit
        cumsum += transmit

        if new_queue[j] > 1e-10:  # 浮点精度容差
            first_non_zero = j
            break

    if first_non_zero == -1:
        non_zero_indices = np.where(new_queue > 1e-10)[0]
        first_non_zero = non_zero_indices[0] if len(non_zero_indices) > 0 else len(queues) - 1
    return new_queue, first_non_zero



def step(state, action, user_cell_index, channel_gain_matrix, user_lambda1, user_lambda2, user_lambda3,
         traffic_list1,
         traffic_list2, traffic_list3, t,
         traffic1, traffic2, traffic3,rr,rr1):
    H = channel_gain_matrix
    P_max = 350
    BW = 200 * 10 ** 6
    N_subchannel = 20
    P_n = 10 ** (-17.4) * BW / (10 ** 3)
    user_h_s = state[60 * 19:60 * 20]
    time_h_s1 = state[60 * 2:60 * 3]
    time_h_s2 = state[60 * 7:60 * 8]
    time_h_s3 = state[60 * 11:60 * 12]
    R_a1 = state[60 * 4:60 * 5] * (10 ** 4)
    R_a2 = state[60 * 9:60 * 10] * (10 ** 4)
    R_a3 = state[60 * 13:60 * 14] * (10 ** 4)
    traffic_ss = state[60 * 14:60 * 15]
    traffic_sums1 = np.sum(traffic1, axis=1)
    traffic_sums2 = np.sum(traffic2, axis=1)
    traffic_sums3 = np.sum(traffic3, axis=1)
    uu_index = np.where((traffic_sums1 + traffic_sums2 + traffic_sums3) > 1e-5)[0]
    uu_cell = user_cell_index[uu_index]

    user_cell_index_filter = np.unique(uu_cell)
    user_cell_index_complete = np.ones(40) * 40
    user_cell_index_complete[0:len(user_cell_index_filter)] = user_cell_index_filter
    #print(action)
    bb = np.argpartition(action, -8)[-8:]#action[0:8].astype(int)
    #bb = action.argsort()[-8:]
    #bb = action[0:8].astype(int)
    #bb = np.unique(action.argmax(axis=1))
    beam_schedule_temp = user_cell_index_complete[bb]
    beam_schedule = beam_schedule_temp[beam_schedule_temp != 40].astype(int)

    if len(beam_schedule)>0:
        P = P_max / len(beam_schedule) * np.ones(len(beam_schedule))   
    else:
        P = np.ones(8)  
    user_index = []

    for i in range(len(beam_schedule)):
        user_index.append(uu_index[np.where(uu_cell == beam_schedule[i])[0]])
    user = np.concatenate(user_index)
    I = [[] for _ in range(len(beam_schedule))]
    S = [[] for _ in range(len(beam_schedule))]
    Capacity = [[] for _ in range(len(beam_schedule))]
    R = [[] for _ in range(len(beam_schedule))]
    for i in range(len(beam_schedule)):
        for j in range(len(user_index[i])):
            H_values = H[beam_schedule, uu_index[np.where(uu_cell == beam_schedule[i])[0]][j]]
            S[i].append(
                H[beam_schedule[i]][uu_index[np.where(uu_cell == beam_schedule[i])[0]][j]] * P[i] / N_subchannel)
            I[i].append(np.sum(H_values * P / N_subchannel) - S[i][j])
            Capacity[i].append(np.log2(1 + S[i][j] / (P_n / N_subchannel + I[i][j])))

    R_aa1 = t * R_a1 / (t + 1) + 1e-8
    R_aa2 = t * R_a2 / (t + 1) + 1e-8
    R_aa3 = t * R_a3 / (t + 1) + 1e-8
    RR = np.zeros(60)
    RR_d = np.zeros(60)
    for i in range(len(beam_schedule)):
        subchannel_res = N_subchannel
        while subchannel_res > 0 and (
                np.sum(
                    traffic_sums1[user_index[i]] + traffic_sums2[user_index[i]] + traffic_sums3[user_index[i]])) > 1e-2:

            w1 = 20 * time_h_s1[user_index[i]] * traffic_sums1[user_index[i]] * np.minimum(
                subchannel_res * 10 ** (-5) * BW / N_subchannel * np.array(Capacity[i]),
                traffic_sums1[user_index[i]]) / R_aa1[user_index[i]]
            w2 = 30 * time_h_s2[user_index[i]] * traffic_sums2[user_index[i]] * np.minimum(
                subchannel_res * 10 ** (-5) * BW / N_subchannel * np.array(Capacity[i]),
                traffic_sums2[user_index[i]]) / R_aa2[user_index[i]]
            w3 = traffic_sums3[user_index[i]] * np.minimum(
                subchannel_res * 10 ** (-5) * BW / N_subchannel * np.array(Capacity[i]),
                traffic_sums3[user_index[i]]) / R_aa3[user_index[i]]
            if np.sum(w1) + np.sum(w2) > 1e-5:
                if np.max(w1) > np.max(w2):
                    user_i = np.argmax(w1)
                    R_temp = min(subchannel_res * 10 ** (-5) * BW / N_subchannel * Capacity[i][user_i],
                                 traffic_sums1[user_index[i][user_i]])
                    traffic1[user_index[i][user_i], :], first_time1 = update_queue_single(
                        traffic1[user_index[i][user_i], :],
                        R_temp)
                    subchannel_res = subchannel_res - math.ceil(
                        R_temp / (10 ** (-5) * BW / N_subchannel * Capacity[i][user_i]))
                    time_h_s1[user_index[i][user_i]] = (30 - first_time1) / 30
                    R_aa1[user_index[i][user_i]] = (t * R_a1[user_index[i][user_i]] + R_temp) / (t + 1) + 1e-8
                    RR_d[user_index[i][user_i]] = RR_d[user_index[i][user_i]] + R_temp
                else:
                    user_i = np.argmax(w2)
                    R_temp = min(subchannel_res * 10 ** (-5) * BW / N_subchannel * Capacity[i][user_i],
                                 traffic_sums2[user_index[i][user_i]])
                    traffic2[user_index[i][user_i], :], first_time2 = update_queue_single(
                        traffic2[user_index[i][user_i], :],
                        R_temp)
                    subchannel_res = subchannel_res - math.ceil(
                        R_temp / (10 ** (-5) * BW / N_subchannel * Capacity[i][user_i]))
                    time_h_s2[user_index[i][user_i]] = (10 - first_time2) / 10
                    R_aa2[user_index[i][user_i]] = (t * R_a2[user_index[i][user_i]] + R_temp) / (t + 1) + 1e-8
                    RR_d[user_index[i][user_i]] = RR_d[user_index[i][user_i]] + R_temp
            else:

                user_i = np.argmax(w3)
                R_temp = min(subchannel_res * 10 ** (-5) * BW / N_subchannel * Capacity[i][user_i],
                             traffic_sums3[user_index[i][user_i]])

                traffic3[user_index[i][user_i], :], first_time3 = update_queue_single(
                    traffic3[user_index[i][user_i], :],
                    R_temp)
                subchannel_res = subchannel_res - math.ceil(
                    R_temp / (10 ** (-5) * BW / N_subchannel * Capacity[i][user_i]))

                time_h_s3[user_index[i][user_i]] = (100 - first_time3) / 100
                R_aa3[user_index[i][user_i]] = (t * R_a3[user_index[i][user_i]] + R_temp) / (t + 1) + 1e-8

            RR[user_index[i][user_i]] = RR[user_index[i][user_i]] + R_temp
            traffic_sums1 = np.sum(traffic1, axis=1)
            traffic_sums2 = np.sum(traffic2, axis=1)
            traffic_sums3 = np.sum(traffic3, axis=1)
    R_a1 = R_aa1 / (10 ** 4)
    R_a2 = R_aa2 / (10 ** 4)
    R_a3 = R_aa3 / (10 ** 4)

    weights1 = np.arange(300, 9, -10)  # 生成 [0, 10, 20, ..., 290]
    weights2 = np.arange(100, 9, -10)  # 生成 [0, 10, 20, ..., 100]
    weights3 = np.arange(1000, 9, -10)
    
    traffic_beam = np.zeros(60)
    traffic_beam[uu_index] = ((R_aa1[uu_index] + R_aa2[uu_index] + R_aa3[uu_index]) * (
            t + 1))  # np.sum(traffic1[:, 1:30], axis=1) +np.sum(traffic2[:, 1:10], axis=1)+np.sum(traffic3[:, 1:100], axis=1)
    # lambda_beam  =  user_lambda1*29+user_lambda2*9+user_lambda3*99
    lambda_beam = np.zeros(60)
    lambda_beam[uu_index] = traffic_ss[uu_index] * 10 ** 4 + (
            user_lambda1[uu_index] + user_lambda2[uu_index] + user_lambda3[uu_index]) * t
    s_user = np.divide(traffic_beam[uu_index],
                       lambda_beam[uu_index],
                       where=(lambda_beam[uu_index] != 0),
                       out=np.zeros_like(lambda_beam[uu_index]))
    s_user = np.clip(s_user, 0, 1)
    #r1 = np.sum(RR) / (np.sum(user_lambda1) + np.sum(user_lambda2) + np.sum(user_lambda3)) / (np.sum(P) / P_max)
    #r2 = np.sum(RR_d) / (np.sum(user_lambda1) + np.sum(user_lambda2))
    r1=np.sum(RR) /(np.sum(P))
    r2=np.sum(RR_d)
    r3 = -np.std(s_user)  # 1-np.var(s_user)
    r4=np.sum(RR)
    #r4 = np.sum(RR) / (np.sum(user_lambda1) + np.sum(user_lambda2) + np.sum(user_lambda3))
    rr=rr+RR
    #r1=np.sum(RR) /(np.sum(P))
    #r2=np.sum(RR_d)
    #r3 = -np.std(s_user)  # 1-np.var(s_user)
    #r4=np.sum(RR)
    traffic1 = np.hstack((traffic1[:, 1:30], traffic_list1[:, np.newaxis]))
    traffic2 = np.hstack((traffic2[:, 1:10], traffic_list2[:, np.newaxis]))
    traffic3 = np.hstack((traffic3[:, 1:100], traffic_list3[:, np.newaxis]))
    rr1=traffic_list1+traffic_list2+traffic_list3+rr1
    time_h_s1 = np.minimum(time_h_s1 + 1 / 30, 1)
    time_h_s2 = np.minimum(time_h_s2 + 1 / 10, 1)
    if t + 1 == 200:
        done = True

    else:
        done = False
    reward = 0.4 * r1 + 0.4 * r2 + 0.2 * r3
    user_h_s[user_h_s < 0.5] = user_h_s[user_h_s < 0.5] - 1 / 400
    user_h_s[user_h_s >= 0.5] = user_h_s[user_h_s >= 0.5] + 1 / 400
    user_h_s = np.clip(user_h_s, a_min=0, a_max=1.0)
    traffic1[user_h_s < 1e-5, :] = 0
    traffic2[user_h_s < 1e-5, :] = 0
    traffic3[user_h_s < 1e-5, :] = 0
    mask_h = (user_h_s > 1e-5)
    traffic_sums1 = np.sum(traffic1, axis=1)
    traffic_sums2 = np.sum(traffic2, axis=1)
    traffic_sums3 = np.sum(traffic3, axis=1)
        
    delay1 = np.divide(np.sum(traffic1[:, 0:29] * weights1[1:30], axis=1),
                           traffic_sums1, where=traffic_sums1 != 0, out=np.zeros_like(traffic_sums1)) 
    delay2 = np.divide(np.sum(traffic2[:, 0:9] * weights2[1:10], axis=1),
                           traffic_sums2, where=traffic_sums2 != 0, out=np.zeros_like(traffic_sums2)) 
    delay3 = np.divide(np.sum(traffic3[:, 0:99] * weights3[1:100], axis=1),
                           traffic_sums3, where=traffic_sums3 != 0, out=np.zeros_like(traffic_sums3)) 
    traffic_s1 = np.sum(traffic1, axis=1) / np.sum(user_lambda1)
    traffic_s2 = np.sum(traffic2, axis=1) / np.sum(user_lambda2)
    traffic_s3 = np.sum(traffic3, axis=1) / np.sum(user_lambda3)

    H_state = state[60 * 15:60 * 16]
    state_rest = state[60 * 16:60 * 19]
    next_state = np.concatenate(
        [traffic_s1, mask_h * delay1 / 300, np.where(user_h_s == 0, 1, time_h_s1), mask_h * arrival_lambda1 / (10 ** 3),
         mask_h * R_a1,
         traffic_s2, mask_h * delay2 / 100, np.where(user_h_s == 0, 1, time_h_s2), mask_h * arrival_lambda2 / (10 ** 3),
         mask_h * R_a2,
         traffic_s3, mask_h * delay3 / 1000, mask_h * arrival_lambda3 / (10 ** 3), mask_h * R_a3,
         traffic_ss * mask_h, H_state * mask_h, state_rest, user_h_s])
    return traffic1, traffic2, traffic3, next_state, reward, r1, r2, r3, r4, rr,rr1,done


#torch.set_num_threads(5)
N_beam = 8
gamma = 0.99
lam = 0.95
num_episodes = 1000
low_steps_per_high = 5
user_location = np.load("user_location1.npy")  # 经度, 纬度
resolution = 5
geo_to_h3_vec = np.vectorize(lambda lat, lng: h3.latlng_to_cell(lat, lng, resolution))
# 获取所有用户的 H3 索引 (向量化计算)
user_h3_indices = geo_to_h3_vec(user_location[:, 1], user_location[:, 0])
unique_h3_indices, user_cell_index = np.unique(user_h3_indices, return_inverse=True)
cell_location = np.array(np.vectorize(h3.cell_to_latlng)(unique_h3_indices)).T
satellite_location = np.load("satellite_location.npy")  # 纬度, 经度
traffic_list1 = np.load("traffic1.npy")
traffic_list2 = np.load("traffic2.npy")
traffic_list3 = np.load("traffic3.npy")
user_num = len(user_location)
cell_num = len(cell_location)
user_lambda1 = np.load("user_lambda1.npy")
user_lambda2 = np.load("user_lambda2.npy")
user_lambda3 = np.load("user_lambda3.npy")
user_t = np.load("user_t1.npy")
agent = SACAgent(20 * 60, 40)
agent.actor.load_state_dict(torch.load("model_actor_SAC_350.pth", map_location=torch.device('cpu')))

# 加载 Critic 网络参数
#agent.critic.load_state_dict(torch.load("model_critic_SAC_2.pth", map_location=torch.device('cpu')))
#agent.target_critic.load_state_dict(torch.load("model_critic_SAC_2.pth", map_location=torch.device('cpu')))
#agent.optim_alpha.load_state_dict(torch.load("model_optim_alpha_SAC_2.pth", map_location=torch.device('cpu')))
 
            
channel_gain_matrix = channel_gain_T(satellite_location, user_location, user_num, cell_num)
user_index_T = []
user_h_s_T = []
for T_s in range(30):
    user_t0 = user_t[:, 200 * T_s]
    user_index_T.append(np.where(user_t0 != 0)[0])
for T_s in range(29):
    user_t1 = user_t[:, 200 * T_s:200 * (T_s + 1)].copy()
    for i in range(user_t1.shape[0]):
        row = user_t[i][200 * T_s:200 * (T_s + 1)]
        zero_positions = np.where(row == 0)[0]

        if zero_positions.size == 0:
            # 当前行全是1，没有0，不做替换
            continue

        first_zero_idx = zero_positions[0]
        N1 = first_zero_idx
        replacement = np.linspace(N1, 1, N1) / 200  # 从N/200到1/200
        user_t1[i, :N1] = replacement  # 替换前

    user_h_s_T.append(user_t1[:, 0])
'''
for T_s in range(29):
    user_t0 = user_t[:, 200 * T_s]
    user_t2 = user_t[:, 200 * (T_s + 1)]
    user_h_s = np.zeros(user_num)
    user_h_s = np.where((user_t0 != 0) & (user_t2 == 0), 1, user_h_s)
    user_h_s_T.append(user_h_s)
'''
user_h_s = np.zeros(user_num)
user_h_s_T.append(user_h_s)
MAX_EPISODE = 50
rr=np.zeros(60)
MAX_T = 30
Reward = []
Reward0 = []
Reward1 = []
Reward2 = []
Reward3 = []
Reward4 = []
Reward0 = []
loss = []
for episode in range(MAX_EPISODE):
    if episode % 1000 == 0:  # 每1000次回收一次
        gc.collect()
    Reward_e = []
    Reward1_e = []
    Reward2_e = []
    Reward3_e = []
    Reward4_e = []

    for T_s in range(0, 10):  # 2s
        if T_s == 0:
            traffic1 = np.load("init_traffic1.npy")
            traffic2 = np.load("init_traffic2.npy")
            traffic3 = np.load("init_traffic3.npy")
            rr1=np.sum(traffic1, axis=1)+np.sum(traffic2, axis=1)+np.sum(traffic3, axis=1)
        weights1 = np.arange(300, 9, -10)  # 生成 [0, 10, 20, ..., 290]
        weights2 = np.arange(100, 9, -10)  # 生成 [0, 10, 20, ..., 100]
        weights3 = np.arange(1000, 9, -10)
        traffic_sums1 = np.sum(traffic1, axis=1)
        traffic_sums2 = np.sum(traffic2, axis=1)
        traffic_sums3 = np.sum(traffic3, axis=1)
        uu_index = np.where((traffic_sums1 + traffic_sums2 + traffic_sums3) > 0)[0]
        uu_cell = user_cell_index[uu_index]
        user_cell_index_filter = np.unique(uu_cell)
        add = np.ones(user_num)
        last_time1 = np.ones(user_num)
        last_time2 = np.ones(user_num)

        delay1 = np.divide(np.sum(traffic1[:, 0:29] * weights1[1:30], axis=1),
                           traffic_sums1, where=traffic_sums1 != 0, out=np.zeros_like(traffic_sums1)) / 300
        delay2 = np.divide(np.sum(traffic2[:, 0:9] * weights2[1:10], axis=1),
                           traffic_sums2, where=traffic_sums2 != 0, out=np.zeros_like(traffic_sums2)) / 100
        delay3 = np.divide(np.sum(traffic3[:, 0:99] * weights3[1:100], axis=1),
                           traffic_sums3, where=traffic_sums3 != 0, out=np.zeros_like(traffic_sums3)) / 1000
        # traffic_sca = np.sum(traffic, axis=1)
        traffic_s1 = np.sum(traffic1, axis=1) / np.sum(user_lambda1)
        traffic_s2 = np.sum(traffic2, axis=1) / np.sum(user_lambda2)
        traffic_s3 = np.sum(traffic3, axis=1) / np.sum(user_lambda3)

        add[uu_index] = np.array([np.where(user_cell_index_filter == a)[0][0] / 40 for a in uu_cell])

        H_s = np.zeros(user_num)
        arrival_lambda1 = np.zeros(user_num)
        arrival_lambda2 = np.zeros(user_num)
        arrival_lambda3 = np.zeros(user_num)

        arrival_lambda1[uu_index] = user_lambda1[uu_index] / (10 ** 3)
        arrival_lambda2[uu_index] = user_lambda2[uu_index] / (10 ** 3)
        arrival_lambda3[uu_index] = user_lambda3[uu_index] / (10 ** 3)
        # 向量化获取 H_s (仅对有效用户赋值)

        H_s[uu_index] = channel_gain_matrix[T_s][
                            user_cell_index[uu_index], np.arange(user_num)[uu_index]] / 10 ** (-12)
        # 2. 获取经纬度信息 cell_lat_s 和 cell_lon_s
        cell_lat_s = np.zeros(user_num)
        cell_lon_s = np.zeros(user_num)
        beam_usage = np.zeros(user_num)
        R_a1 = np.zeros(user_num)
        R_a2 = np.zeros(user_num)
        R_a3 = np.zeros(user_num)
        R_a1[uu_index] = 1e-8
        R_a2[uu_index] = 1e-8
        R_a3[uu_index] = 1e-8
        # 向量化获取经纬度
        cell_lat_s[uu_index] = cell_location[user_cell_index[uu_index], 0] / satellite_location[T_s][0]
        cell_lon_s[uu_index] = cell_location[user_cell_index[uu_index], 1] / satellite_location[T_s][1]

        user_h_s = user_h_s_T[T_s] / 2
        # user_h_s = np.where((user_t0 != 0) & (user_t1 == 0), 1, user_h_s)
        state = np.concatenate(
            [traffic_s1, delay1, last_time1, arrival_lambda1, R_a1,
             traffic_s2, delay2, last_time2, arrival_lambda2, R_a2,
             traffic_s3, delay3, arrival_lambda3, R_a3, traffic_s1 + traffic_s2 + traffic_s3,
             H_s, cell_lat_s, cell_lon_s, add, user_h_s])

        states, actions, powers, scales, log_probs, rewards, reward1s, reward2s, reward3s, reward4s, dones, values = [], [], [], [], [], [], [], [], [], [], [], []
        t = 0
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _ = agent.actor(state_tensor)
            states.append(state.astype(np.float32))
            actions.append(action.cpu().numpy()[0])


            # 转 numpy，彻底断开梯度
            #states.append(state.astype(np.float32))  # 如果 state 已经是 np.array
            #actions.append(action.cpu().numpy()[0])
            #print(action)
            traffic1, traffic2, traffic3, state, reward, reward1, reward2, reward3, reward4, rr,rr1,done = step(
                state,
                actions[-1],
                user_cell_index,
                channel_gain_matrix[T_s],
                user_lambda1,
                user_lambda2,
                user_lambda3,
                traffic_list1[:, 200 * T_s + t],
                traffic_list2[:, 200 * T_s + t],
                traffic_list3[:, 200 * T_s + t],
                t, traffic1, traffic2, traffic3,rr,rr1
            )
            # agent.replay_buffer.push(states[-1], actions[-1], reward*10, state, done)
            '''
            if len(agent.replay_buffer) >= 20000:  # MEMORY_CAPACITY
                if t % 5 == 0:
                    agent.learn()
            '''
            rewards.append(reward)
            reward1s.append(reward1)
            reward2s.append(reward2)
            reward3s.append(reward3)
            reward4s.append(reward4)
            dones.append(done)

            if done:
                break

            t += 1
        Reward_e.append(sum(rewards) / 200)
        Reward1_e.append(sum(reward1s) / 200)
        Reward2_e.append(sum(reward2s) / 200)
        Reward3_e.append(sum(reward3s) / 200)
        Reward4_e.append(sum(reward4s) / 200)

        print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(rewards) / 200:.5f}")
        #print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(reward1s) / 200:.5f}")
        #print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(reward2s) / 200:.5f}")
        #print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(reward3s) / 200:.5f}")
        #print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(reward4s) / 200:.5f}")
    Reward.append(np.sum(Reward_e) / 10)
    Reward1.append(np.sum(Reward1_e) / 10)
    Reward2.append(np.sum(Reward2_e) / 10)
    Reward3.append(np.sum(Reward3_e) / 10)
    Reward4.append(np.sum(Reward4_e) / 10)
    
    np.savetxt('reward_3e-5_SAC0.txt', Reward)
    np.savetxt('reward1_3e-5_SAC0.txt', Reward1)
    np.savetxt('reward2_3e-5_SAC0.txt', Reward2)
    np.savetxt('reward3_3e-5_SAC0.txt', Reward3)
    np.savetxt('reward4_3e-5_SAC0.txt', Reward4)



