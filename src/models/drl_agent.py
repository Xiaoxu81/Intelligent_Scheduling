import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, max_queue_size, task_feat_dim=9, system_feat_dim=6, resource_feat_dim=5, weight_feat_dim=4, num_strategies=12):
        super(ActorCritic, self).__init__()
        
        self.num_strategies = num_strategies
        
        # 任务特征嵌入层
        self.task_embed = nn.Sequential(
            nn.Linear(task_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 综合特征处理层
        combined_dim = 64 + system_feat_dim + resource_feat_dim + weight_feat_dim
        self.common = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor: 输出策略选择概率 (12种策略组合)
        self.actor = nn.Linear(64, num_strategies)
        
        # Critic: 输出状态价值
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        # obs['tasks']: (batch, max_queue_size, task_feat_dim)
        # obs['system']: (batch, 6), obs['resources']: (batch, R, 5), obs['weights']: (batch, 4)
        
        tasks = obs['tasks']
        system = obs['system']
        resources = obs['resources']
        weights = obs['weights']
        
        # 1. 任务特征嵌入与聚合
        t_embeds = self.task_embed(tasks) # (batch, max_queue_size, 32)
        # 同时保留平均负载和极值负载，避免少量长任务被平均值掩盖。
        t_mean = torch.mean(t_embeds, dim=1) # (batch, 32)
        t_max = torch.max(t_embeds, dim=1).values # (batch, 32)
        t_agg = torch.cat([t_mean, t_max], dim=1) # (batch, 64)
        
        resource_agg = torch.mean(resources, dim=1)
        combined = torch.cat([t_agg, system, resource_agg, weights], dim=1)
        
        # 3. 提取特征
        x = self.common(combined)
        
        # 4. 输出
        probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return probs, value

class PPOAgent:
    def __init__(self, max_queue_size=10, task_feat_dim=9, system_feat_dim=6, resource_feat_dim=5, weight_feat_dim=4, num_strategies=12, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.model = ActorCritic(max_queue_size, task_feat_dim, system_feat_dim, resource_feat_dim, weight_feat_dim, num_strategies)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model_old = ActorCritic(max_queue_size, task_feat_dim, system_feat_dim, resource_feat_dim, weight_feat_dim, num_strategies)
        self.model_old.load_state_dict(self.model.state_dict())
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_queue_size = max_queue_size
        self.num_strategies = num_strategies
        self.MseLoss = nn.MSELoss()

    @staticmethod
    def _obs_to_tensor(obs):
        return {
            "tasks": torch.FloatTensor(obs["tasks"]).unsqueeze(0),
            "system": torch.FloatTensor(obs.get("system", [0.0] * 6)).unsqueeze(0),
            "resources": torch.FloatTensor(obs.get("resources", np.zeros((1, 5), dtype=np.float32))).unsqueeze(0),
            "weights": torch.FloatTensor(obs.get("weights", [0.25] * 4)).unsqueeze(0),
        }

    def select_action(self, obs):
        """选择策略组合 (0-11 对应 C01-C12)"""
        obs_t = self._obs_to_tensor(obs)
        
        with torch.no_grad():
            probs, value = self.model_old(obs_t)
            
        m = Categorical(probs)
        strategy_idx = m.sample()
        return strategy_idx.item(), m.log_prob(strategy_idx).item(), value.item()
    
    def get_strategy_probs(self, obs):
        """获取各策略的选择概率，用于分析"""
        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            probs, _ = self.model_old(obs_t)
        return probs.squeeze().numpy()

    def update(self, memory):
        # 转换内存数据为 Tensor
        old_states_tasks = torch.FloatTensor(np.array([m['obs']['tasks'] for m in memory]))
        old_states_system = torch.FloatTensor(np.array([m['obs']['system'] for m in memory]))
        old_states_resources = torch.FloatTensor(np.array([m['obs']['resources'] for m in memory]))
        old_states_weights = torch.FloatTensor(np.array([m['obs']['weights'] for m in memory]))
        old_actions = torch.LongTensor(np.array([m['action'] for m in memory]))
        old_logprobs = torch.FloatTensor(np.array([m['log_prob'] for m in memory]))
        
        # 计算回报和优势
        rewards = []
        discounted_reward = 0
        for m in reversed(memory):
            if m['done']:
                discounted_reward = 0
            discounted_reward = m['reward'] + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.FloatTensor(rewards)
        # 归一化奖励
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-7)

        # 迭代更新 K 次
        for _ in range(self.K_epochs):
            # 获取当前模型的概率和价值
            probs, state_values = self.model({
                "tasks": old_states_tasks,
                "system": old_states_system,
                "resources": old_states_resources,
                "weights": old_states_weights,
            })
            
            dist = Categorical(probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # 计算比例 (r_t)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算优势 (Advantages)
            state_values_flat = state_values.squeeze(-1)
            advantages = rewards - state_values_flat.detach()

            # PPO 损失函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values_flat, rewards) - 0.01 * dist_entropy
            
            # 梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 同步旧模型
        self.model_old.load_state_dict(self.model.state_dict())

    def pretrain_bc(self, expert_data, epochs=50):
        """
        行为克隆 (Behavior Cloning)：利用专家数据预训练 Actor
        expert_data: [(obs, action), ...]
        """
        print(f"Starting Behavior Cloning pre-training for {epochs} epochs...")
        states_tasks = torch.FloatTensor(np.array([d[0]['tasks'] for d in expert_data]))
        states_system = torch.FloatTensor(np.array([d[0]['system'] for d in expert_data]))
        states_resources = torch.FloatTensor(np.array([d[0]['resources'] for d in expert_data]))
        states_weights = torch.FloatTensor(np.array([d[0]['weights'] for d in expert_data]))
        actions = torch.LongTensor(np.array([d[1] for d in expert_data]))

        for epoch in range(epochs):
            probs, _ = self.model({
                "tasks": states_tasks,
                "system": states_system,
                "resources": states_resources,
                "weights": states_weights,
            })
            
            # probs 已经经过 softmax，因此使用负对数似然进行行为克隆。
            loss = nn.functional.nll_loss(torch.log(probs + 1e-8), actions)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"BC Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        self.model_old.load_state_dict(self.model.state_dict())
