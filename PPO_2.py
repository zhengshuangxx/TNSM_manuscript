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

# 记录开始时间

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

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
            nn.Linear(64, action_dim),
            nn.ReLU(),
        )
       
    def forward(self, state):
        enc = self.net(state)  # [B, 128]

        logits = self.beam_index(enc)
        actions = gumbel_top_k_sampling(logits)
        # print(log_probs)

        log_prob_dis = plackett_luce_log_prob(logits, actions)
        total_log_prob = log_prob_dis 
        return actions.detach(),  total_log_prob

    def evaluate_actions(self, state, actions):
        enc = self.net(state)  # [B, 128]

        logits = self.beam_index(enc)
        log_prob_dis = plackett_luce_log_prob(logits, actions)

        # log_dis_probs = torch.log(probs + 1e-8)  # 防止 log(0)
        # dis_entropy = -torch.sum(probs * log_dis_probs, dim=1)  # [B] 每个样本的熵

        entropy = compute_entropy_for_known_actions_vectorized(logits, actions)
        total_log_prob = log_prob_dis 
        total_entropy = entropy 
        return total_log_prob, total_entropy


def plackett_luce_log_prob(logits, actions):
    B, K = actions.shape
    log_probs = []
    mask = torch.zeros_like(logits, dtype=torch.bool)  # 初始掩码
    actions = actions.long()
    for i in range(K):
        # 创建新掩码（非原地）
        if i > 0:
            mask = mask.scatter(1, actions[:, i - 1:i], True)  # 非原地scatter

        # 应用掩码
        masked_logits = logits.masked_fill(mask, -1e6)  # 改用-1e6更稳定
        log_p = F.log_softmax(masked_logits, dim=1)
        log_probs.append(log_p.gather(1, actions[:, i:i + 1]))
    return torch.cat(log_probs, dim=1).sum(dim=1)


def gumbel_top_k_sampling(logits):
    U = torch.rand(logits.shape, device=device).clamp(min=1e-6, max=0.999)
    gumbel_noise = -torch.log(-torch.log(U))
    # Gumbel噪声
    perturbed_logits = (logits + gumbel_noise) / 1

    # perturbed_logits=perturbed_logits - perturbed_logits.mean(dim=-1, keepdim=True)
    log_probs = F.log_softmax(perturbed_logits, dim=-1)
    _, topk_indices = perturbed_logits.topk(8, dim=-1)
    return topk_indices


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


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim,
                             512)  # 展平后通过全连接层
        self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # 输出状态价
        self.bn1 = nn.LayerNorm(256)  # 用于对 fc1 的输出进行归一化
        self.bn2 = nn.LayerNorm(128)  # 用于对 fc2 的输出进行归一化
        # self.bn3 = nn.LayerNorm(128)

        # Dropout 层
        # self.dropout = nn.Dropout(0.2)  # 使用 50% dropout 防止过拟合
        # nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        # x = F.relu(self.fc3(x))
        # x = self.bn3(x)
        # x = self.dropout(x)  # 添加 Dropout
        x = self.fc3(x)
        return x


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]  # Append 0 for the terminal state
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.append(gae)  # Append instead of insert

    advantages.reverse()  # Reverse at the end to maintain the correct order
    # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    # print(advantages)
    return advantages, returns


def compute_gae1(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + v for adv, v in zip(advantages, values[:-1])]

    return advantages, returns


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-5):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=3e-5)  # 3e-4   1e-4 3e-5(clip0.95 3e-5 0.99)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=3e-5)  # 5e-4 5e-4  1e-4

    def update(self, states, actions,log_probs_old, returns, advantages, step, loss,
               clip_eps=0.2, epochs=5, batch_size=50, action_dim=52):  # 5
        # 确保batch_size能整除数据量
        assert len(states) % batch_size == 0, "数据量应能被batch_size整除"

        # 转换为PyTorch张量
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        # 创建数据集
        dataset = torch.utils.data.TensorDataset(
            states, actions, log_probs_old, returns, advantages)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        initial_std = torch.exp(torch.tensor(-2))  # nn.Parameter(torch.zeros(action_dim))  # 可训练的对数标准差
        decay_rate = 0.995
        min_std = torch.exp(torch.tensor(-4.5))

        # std = torch.ones(action_dim) * (decay_rate ** step * initial_std)
        # std = torch.clamp(std, min_std, initial_std).to(device)
        # log_std = torch.ones(action_dim) * (-2)
        # std = torch.exp(log_std).to(device)
        tau = max(0.5, 1 * (0.95 ** step))
        for _ in range(epochs):
            for batch in data_loader:
                batch_states, batch_actions,batch_log_probs, batch_returns, batch_advantages = batch

                # Actor更新
                log_probs_new, entropy = self.actor.evaluate_actions(batch_states, batch_actions)

                ratio = (log_probs_new - batch_log_probs).exp()
                # max_ratio = 20  # 设定最大允许的 ratio，防止更新过大

                # clip_eps = np.clip(clip_eps * 0.9995 ** step, 0.1, 0.2)
                clip_eps = 0.2
                clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                # batch_advantages_n = (batch_advantages - torch.mean(batch_advantages)) / (torch.std(batch_advantages) + 1e-8)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                self.optim_actor.zero_grad()
                (policy_loss - 0.1 * entropy.mean()).backward()
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optim_actor.step()

                # Critic更新
                # Critic更新
                # (values - batch_returns).pow(2).mean()
                values = self.critic(batch_states).squeeze()
                # print(values)
                # print(batch_returns)
                # print(values)
                # print(batch_returns)
                value_loss = F.mse_loss(values, batch_returns)  #
                # 先创建 MSELoss 的实例
                # mse_loss = torch.nn.MSELoss()

                # 再传入预测值和目标值计算损失
                # value_loss = mse_loss(values, batch_returns)
                # F.mse_loss(values, batch_returns, delta=1.0)
                self.optim_critic.zero_grad()
                value_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optim_critic.step()
                '''ok
                values = self.critic(batch_states).squeeze()
                value_loss = (values - batch_returns).pow(2).mean()

                self.optim_critic.zero_grad()
                value_loss.backward()
                self.optim_critic.step()
                '''
        print(entropy.mean().item())
        print(policy_loss.item())
        # print(entropy)
        # 假设 model 是训练好的 PyTorch 模型
        '''
        if step % 50 == 0:
            torch.save(self.actor.state_dict(), 'model_actor_PPO_350.pth')
            torch.save(self.critic.state_dict(), 'model_critic_PPO_350.pth')
        '''
        print(value_loss.item())


def process_action(raw_action, user_cell_index_filter, cell_location):
    # 1. 坐标解码 (映射到实际地理范围)

    beam_lat = 39.9 + 0.5 * raw_action[0:4]  # 北京周边±0.5度

    beam_lon = 116.4 + 0.5 * raw_action[4:8]

    # 2. 寻找最近的有效蜂窝
    beam_centers = cell_location[user_cell_index_filter]
    beam_schedule = []
    for lat, lon in zip(beam_lat, beam_lon):
        distances = [geodesic((lat, lon), (c[0], c[1])).km for c in beam_centers]
        beam_schedule.append(user_cell_index_filter[np.argmin(distances)])

    # 3. 功率分配 (Softmax归一化)
    power_weights = softmax(np.clip(raw_action[16:24], 1e-6, np.inf))
    return beam_schedule, power_weights


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 防溢出
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


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
    '''
    for i in range(cell_num):
        d_sc[i]=np.sqrt((x_sat[i]-x_cell[i])**2+(y_sat[i]-y_cell[i])**2+(z_sat[i]-z_cell[i])**2)
    for i in range(user_num):
        d_su[i]=np.sqrt((x_sat[i]-x_user[i])**2+(y_sat[i]-y_user[i])**2+(z_sat[i]-z_user[i])**2)
        PL[i] = -20 * np.log10(d_su[i]) - 20 * np.log10(f) - 20 * np.log10(
                4 * np.pi * 10 ** 9 / (3 * 10 ** 8)) - PL_o
    for i in range(cell_num):
        for j in range(user_num):
            d_cu[i][j]=np.sqrt((x_cell[i]-x_user[i])**2+(y_cell[i]-y_user[i])**2+(z_cell[i]-z_user[i])**2)
            theta[i][j]=np.acos((d_su[j]**2+d_sc[i]**2-2*d_su[j]*d_sc[i])/(2*d_su[j]*d_sc[i]))
            r[i][j]= 2.07123 * np.sin(theta[i][j]) / np.sin(theta_3dB)
            if r[i][j] == 0:
                G_s[i][j] = Gs_max
            else:
                G_s[i][j] = 10 * np.log10(10 ** (Gs_max / 10) * (
                            jv(1, r[i][j]) / (2 * r[i][j]) + 36 * (jv(3, r[i][j]) / r[i][j] ** 3)) ** 2)
            H[i][j] = 10 ** ((Gr_max + G_s[i][j] + PL[j]) / 10)
    '''
    return H


def channel_gain_T(satellite_location, user_location, user_num, cell_num):
    T_s = 30
    H = []
    for t in range(T_s):
        H.append(channel_gain(satellite_location, user_location, user_num, cell_num, t))
    return H


def update_queue_vectorized(queues, transfer_sizes):
    queues = np.array(queues, dtype=np.float64)
    transfer_sizes = np.array(transfer_sizes, dtype=np.float64)
    num_users, num_packets = queues.shape
    new_queues = queues.copy()
    first_non_zero = np.full(num_users, -1)

    for i in range(num_users):
        T = transfer_sizes[i]
        if T <= 0:
            first_non_zero[i] = 0 if queues[i].any() else -1
            continue

        cumsum = 0
        for j in range(num_packets):
            if cumsum >= T:
                break

            remaining = T - cumsum
            transmit = min(remaining, queues[i, j])
            new_queues[i, j] -= transmit
            cumsum += transmit

            if new_queues[i, j] > 1e-10:  # 浮点精度容差
                first_non_zero[i] = j
                break

        # 如果所有包都未完全传输完T
        if first_non_zero[i] == -1 and queues[i].any():
            first_non_zero[i] = 0

    return new_queues, first_non_zero


def update_queue_vectorized1(queues, transfer_sizes):
    queues = np.array(queues, dtype=np.float64)
    transfer_sizes = np.array(transfer_sizes, dtype=np.float64)
    num_users, num_packets = queues.shape

    # 计算累积和
    cumsum = np.cumsum(queues, axis=1)

    # 找到每个用户能传输的最后一个包的索引
    mask = cumsum <= transfer_sizes[:, None]
    last_idx = np.where(mask.any(axis=1), np.argmax(~mask, axis=1) - 1, num_packets - 1)

    # 处理传输量超过总和的情况
    total_queues = cumsum[:, -1]
    overflow_mask = transfer_sizes > total_queues
    last_idx[overflow_mask] = num_packets - 1

    # --- 核心向量化操作 ---
    rows = np.arange(num_users)[:, None]
    cols = np.arange(num_packets)

    # 1. 已传输部分清零
    clear_mask = cols <= last_idx[:, None]
    queues[clear_mask] = 0

    # 2. 处理最后一个包的部分传输
    valid_mask = (last_idx >= 0) & (last_idx < num_packets - 1)
    remaining = transfer_sizes - np.where(last_idx >= 0, cumsum[rows[:, 0], last_idx], 0)
    remaining = np.where(valid_mask, remaining, 0)
    queues[rows[:, 0], last_idx] = np.maximum(0, queues[rows[:, 0], last_idx] - remaining)

    # 3. 计算每个用户第一个非零索引（无有效包则返回-1）
    non_zero_mask = queues != 0
    first_non_zero = np.argmax(non_zero_mask, axis=1)  # 找到第一个True的位置
    first_non_zero[~non_zero_mask.any(axis=1)] = -1  # 全零行设为-1

    return queues, first_non_zero


def haversine(lat1, lon1, lat2, lon2):
    """ 计算两点间的球面距离（单位：米） """
    R = 6371000  # 地球半径（单位：米）
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    a = np.clip(a, 0, 1)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


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


def find_nearest_location(user_location, locations):
    """
    找到用户位置在 locations 集合中最近的索引

    :param user_location: (纬度, 经度) 用户位置
    :param locations: [(纬度, 经度), ...] 位置集合
    :return: 最近位置的索引
    """
    user_lat, user_lon = user_location
    locs_lat, locs_lon = np.array(locations).T  # 将列表转换为 numpy 数组并拆分成纬度、经度
    distances = haversine(user_lat, user_lon, locs_lat, locs_lon)  # 计算所有距离
    return np.argmin(distances)  # 返回最近位置的索引


def step(state, action, user_cell_index, channel_gain_matrix, user_lambda1, user_lambda2, user_lambda3,
         traffic_list1,
         traffic_list2, traffic_list3, t,
         traffic1, traffic2, traffic3,rr,rr1):
    thr1 = 0
    thr2 = 0
    thr3 = 0
    H = channel_gain_matrix
    P_max = 250
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
    bb = action[0:8].astype(int)
    beam_schedule_temp = user_cell_index_complete[bb]
    beam_schedule = beam_schedule_temp[beam_schedule_temp != 40].astype(int)


    P = P_max / len(beam_schedule) * np.ones(len(beam_schedule))     
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

            w1 = 20 * traffic_sums1[user_index[i]] * np.minimum(
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
                    thr1 = thr1 + R_temp
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
                    thr2 = thr2 + R_temp
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
                thr3 = thr3 + R_temp

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
    #np.sum(RR) /(np.sum(P))
    #r2=np.sum(RR_d)
    #r3 = -np.std(s_user)  # 1-np.var(s_user)
    #r4=np.sum(RR)
    #r4 = np.sum(RR) / (np.sum(user_lambda1) + np.sum(user_lambda2) + np.sum(user_lambda3))
    rr=rr+RR
    #r1=np.sum(RR) /(np.sum(P))
    #r2=np.sum(RR_d)
    #r3 = -np.std(s_user)  # 1-np.var(s_user)
   
    r4=np.sum(RR)
    #r1 = np.sum(traffic1[:, 0])
    #r2 = np.sum(traffic2[:, 0])
    #r3 = np.sum(traffic3[:, 0])
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
    r1 = np.sum(traffic1[:, 0:29] * weights1[1:30])/np.sum(traffic1)
    r2 = np.sum(traffic2[:, 0:9] * weights2[1:10])/np.sum(traffic2)
    r3 = np.sum(traffic3[:, 0:99] * weights3[1:100])/np.sum(traffic3)
    reward = 0.4 * r1 + 0.4 * r2 + 0.2 * r3
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
agent = PPOAgent(20 * 60, 40)
agent.actor.load_state_dict(torch.load("model_actor_PPO_250.pth", map_location=torch.device('cpu')))

# 加载 Critic 网络参数
agent.critic.load_state_dict(torch.load("model_critic_PPO_250.pth", map_location=torch.device('cpu')))
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
        tt_sum1 = np.sum(traffic1) + np.sum(traffic_list1[:, 200 * T_s : 200*(T_s+1)])
        tt_sum2 = np.sum(traffic2) + np.sum(traffic_list2[:, 200 * T_s : 200*(T_s+1)])
        tt_sum3 = np.sum(traffic3) + np.sum(traffic_list3[:, 200 * T_s : 200*(T_s+1)])
        #tt_sum = np.sum(traffic_list1[:, 200 * T_s : 200*(T_s+1)]) + np.sum(traffic_list2[:, 200 * T_s : 200*(T_s+1)]) + np.sum(traffic_list3[:, 200 * T_s : 200*(T_s+1)])
        
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
                action, log_prob = agent.actor(state_tensor)
                value = agent.critic(state_tensor).item()

            # 转 numpy，彻底断开梯度
            states.append(state.astype(np.float32))  # 如果 state 已经是 np.array
            actions.append(action.cpu().numpy()[0])
            log_probs.append(float(log_prob))  # 保证是纯 float

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

            rewards.append(reward)
            reward1s.append(reward1)
            reward2s.append(reward2)
            reward3s.append(reward3)
            reward4s.append(reward4)
            dones.append(done)
            values.append(value)

            if done:
                break

            t += 1

        # 计算优势
        
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        
        # 调用 update 前，确保所有数据是 numpy
        agent.update(
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(log_probs, dtype=np.float32),
            np.array(returns, dtype=np.float32),
            np.array(advantages, dtype=np.float32),
            episode,
            loss
        )
        
        # print(rewards)
        Reward_e.append(sum(rewards) / 200)
        Reward1_e.append(sum(reward1s) /  200)
        Reward2_e.append(sum(reward2s) /  200)
        Reward3_e.append(sum(reward3s) /  200)
        Reward4_e.append(sum(reward4s) / 200)

        print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(rewards) / 200:.5f}")
        print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(reward1s) / 200:.5f}")
        print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(reward2s) / 200:.5f}")
        print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(reward3s) / 200:.5f}")
        print(f"Episode {episode}: Slot{T_s}: Total Reward = {sum(reward4s) / 200:.5f}")
    Reward.append(np.sum(Reward_e) / 10)
    Reward1.append(np.sum(Reward1_e) / 10)
    Reward2.append(np.sum(Reward2_e) / 10)
    Reward3.append(np.sum(Reward3_e) / 10)
    Reward4.append(np.sum(Reward4_e) / 10)
    #np.savetxt('rr_ppo_3e-5.txt', rr)
    #np.savetxt('rr2_ppo_3e-5.txt', rr1)
    #np.savetxt('reward_QoS_3e-5.txt', Reward)
    np.savetxt('reward1_3e-5.txt', Reward1)
    np.savetxt('reward2_3e-5.txt', Reward2)
    np.savetxt('reward3_3e-5.txt', Reward3)
    #np.savetxt('reward4_3e-5.txt', Reward4)



