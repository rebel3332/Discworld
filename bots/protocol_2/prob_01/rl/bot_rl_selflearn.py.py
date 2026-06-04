# bot_rl_selflearn.py
# RL-бот с самообучением (без внешних reward от сервера)

#"""
#1. Запуск обучения (с нуля):
#python bot_rl_selflearn.py
#
#2. Запуск с сохранённой моделью (тест):
#python bot_rl_selflearn.py --model bot_rl_checkpoint_1000.pt --test
#
#3. Продолжить обучение:
#python bot_rl_selflearn.py --model bot_rl_checkpoint_500.pt
#"""



import asyncio
import json
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import websockets
import time

SERVER = "ws://localhost:8000/ws"
PROTOCOL_VERSION = 2

# =========================================================
# REWARD CALCULATOR (на стороне клиента)
# =========================================================

class RewardShaper:
    """
    Вычисляет reward на основе изменений в состоянии игры.
    """
    
    def __init__(self):
        self.prev_hp = 1.0
        self.prev_enemy_dist = 10.0
        self.prev_enemy_count = 0
        self.steps_alive = 0
        self.last_shot_time = 0
        self.damage_taken = 0
        self.shots_fired = 0
        self.score_old = 0
        
    def compute_reward(self, observation, action):
        """
        Вычисляем reward на основе:
        1. Выживание (+0.01 за каждый шаг)
        2. Получение урона (-1.0 за 10% HP)
        3. Приближение к врагам (+0.1 когда враг близко)
        4. Стрельба по врагам (+0.5 если враг в радиусе)
        5. Исследование новых областей (+0.05)
        """
        reward = 0.0
        
        # Текущее состояние
        curr_hp = observation['self']['hp']
        enemy_sensors = observation.get('enemy_sensors', [1.0] * 3)
        wall_sensors = observation.get('wall_sensors', [1.0] * 3)
        score = observation.get('score', 0)
        
        # Минимальная дистанция до врага
        min_enemy_dist = min(enemy_sensors) if enemy_sensors else 10.0
        
        # Количество врагов в поле зрения (дистанция < 1.0)
        curr_enemy_count = sum(1 for d in enemy_sensors if d < 1.0)
        
        # =========================================
        # 1. SURVIVAL REWARD (выживание)
        # =========================================
        reward += 0.001  # маленький бонус за каждый шаг
        
        # =========================================
        # 2. DAMAGE PENALTY (получение урона)
        # =========================================
        hp_delta = curr_hp - self.prev_hp
        if hp_delta < 0:
            damage = abs(hp_delta)
            reward -= damage * 10.0  # большой штраф за урон
            self.damage_taken += damage
        
        # =========================================
        # 3. ENEMY PROXIMITY REWARD
        # =========================================
        # Награда за нахождение врагов рядом (но не слишком близко)
        if min_enemy_dist < 1.0:
            # Враг в поле зрения
            ideal_dist = 0.5  # оптимальная дистанция для стрельбы
            dist_reward = 1.0 - abs(min_enemy_dist - ideal_dist) / ideal_dist
            reward += dist_reward * 0.1
        
        # =========================================
        # 4. SHOOTING REWARD
        # =========================================
        if action.get('shoot', False):
            self.shots_fired += 1
            # Награда за стрельбу когда враг рядом
            if min_enemy_dist < 0.8:
                reward += 0.05
            # Штраф за стрельбу в пустоту
            elif min_enemy_dist > 1.5:
                reward -= 0.01
        
        # =========================================
        # 5. EXPLORATION REWARD
        # =========================================
        # Поощряем движение в новые области
        dx = abs(action.get('dx', 0))
        dy = abs(action.get('dy', 0))
        if dx > 0.1 or dy > 0.1:
            reward += 0.005  # маленький бонус за движение
        
        # =========================================
        # 6. WALL AVOIDANCE REWARD
        # =========================================
        # Штраф за приближение к стенам
        # min_wall_dist = min(wall_sensors) if wall_sensors else 1.0
        # if min_wall_dist < 0.2:
        #     reward -= 0.1  # штраф за близость к стене
        
        # =========================================
        # 7. ENEMY COUNT REWARD
        # =========================================
        # Награда за нахождение нескольких врагов
        if curr_enemy_count > self.prev_enemy_count:
            reward += 0.1  # нашли новых врагов
        
        # =========================================
        # 8. SCORE REWARD
        # =========================================
        # Награда за увеличение счета
        if score > self.score_old:
            reward += (score - self.score_old) * 0.1
            print(f"🎯 Score increased: {score} (+{score - self.score_old})")
        self.score_old = score


        # Сохраняем состояние для следующего шага
        self.prev_hp = curr_hp
        self.prev_enemy_dist = min_enemy_dist
        self.prev_enemy_count = curr_enemy_count
        self.steps_alive += 1
        
        return reward
    
    def reset(self):
        """Сброс после смерти/респавна"""
        if self.prev_hp < 0.1:  # почти умер
            print(f"💀 Died after {self.steps_alive} steps, "
                  f"damage: {self.damage_taken:.2f}, shots: {self.shots_fired}")
        
        self.prev_hp = 1.0
        self.steps_alive = 0
        self.damage_taken = 0
        self.shots_fired = 0


# =========================================================
# NEURAL NETWORK (Policy)
# =========================================================

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim=32, hidden_dim=128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor
        self.actor_mean = nn.Linear(hidden_dim, 3)  # dx, dy, angle_delta
        self.actor_logstd = nn.Parameter(torch.zeros(3))
        self.shoot_head = nn.Linear(hidden_dim, 1)
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        features = self.shared(x)
        
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(self.actor_logstd).clamp(0.05, 0.5)
        shoot_logit = self.shoot_head(features)
        shoot_prob = torch.sigmoid(shoot_logit)
        value = self.critic(features)
        
        return action_mean, action_std, shoot_prob, value
    
    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        mean, std, shoot_prob, value = self.forward(obs_tensor)
        
        if deterministic:
            action = mean
            shoot = shoot_prob > 0.5
            log_prob = 0
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            shoot = torch.rand(1) < shoot_prob
            log_prob = dist.log_prob(action).sum().item()
        
        return (
            action.squeeze().detach().numpy(),
            shoot.item(),
            log_prob,
            value.squeeze().detach().item()
        )
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"💾 Model saved: {path}")
    
    def load(self, path):
        try:
            self.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"📦 Model loaded: {path}")
        except:
            print(f"⚠️  No model found at {path}, starting fresh")


# =========================================================
# REPLAY BUFFER & PPO TRAINER
# =========================================================

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, done, log_prob, value):
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return {
            'obs': np.array([b['obs'] for b in batch]),
            'action': np.array([b['action'] for b in batch]),
            'reward': np.array([b['reward'] for b in batch]),
            'done': np.array([b['done'] for b in batch]),
            'log_prob': np.array([b['log_prob'] for b in batch]),
            'value': np.array([b['value'] for b in batch])
        }
    
    def __len__(self):
        return len(self.buffer)


class PPOTrainer:
    def __init__(self, policy, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
    
    def update(self, buffer, batch_size=64, epochs=4):
        if len(buffer) < batch_size:
            return
        
        for _ in range(epochs):
            batch = buffer.sample(batch_size)
            
            obs = torch.FloatTensor(batch['obs'])
            actions = torch.FloatTensor(batch['action'])
            rewards = torch.FloatTensor(batch['reward'])
            old_log_probs = torch.FloatTensor(batch['log_prob'])
            old_values = torch.FloatTensor(batch['value'])
            
            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Forward pass
            mean, std, shoot_prob, values = self.policy(obs)
            
            # Actor loss (PPO)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * rewards
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = ((values.squeeze() - rewards) ** 2).mean()
            
            # Entropy bonus (для exploration)
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()


# =========================================================
# RL BOT
# =========================================================

class RLBot:
    def __init__(self, model_path=None, train=True):
        self.obs_dim = 32
        self.policy = PolicyNetwork(self.obs_dim)
        
        if model_path:
            self.policy.load(model_path)
        
        self.trainer = PPOTrainer(self.policy) if train else None
        self.buffer = ReplayBuffer() if train else None
        
        self.angle = random.uniform(-math.pi, math.pi)
        self.reward_shaper = RewardShaper()
        
        self.prev_obs = None
        self.train = train
        self.episode_reward = 0
        self.step_count = 0
        
    def preprocess_obs(self, observation):
        """Конвертируем observation в fixed-size vector"""
        wall_sensors = observation.get('wall_sensors', [1.0] * 3)
        enemy_sensors = observation.get('enemy_sensors', [1.0] * 3)
        
        # Дополнительные признаки
        self_data = observation.get('self', {})
        hp = self_data.get('hp', 1.0)
        x = self_data.get('x', 0) / 200.0  # нормализация
        y = self_data.get('y', 0) / 200.0
        angle = self_data.get('angle', 0) / math.pi
        
        # Статистика по врагам
        min_enemy_dist = min(enemy_sensors) if enemy_sensors else 1.0
        avg_enemy_dist = sum(enemy_sensors) / len(enemy_sensors) if enemy_sensors else 1.0
        enemies_nearby = sum(1 for d in enemy_sensors if d < 1.0) / 3.0
        
        # Статистика по стенам
        min_wall_dist = min(wall_sensors) if wall_sensors else 1.0
        avg_wall_dist = sum(wall_sensors) / len(wall_sensors) if wall_sensors else 1.0
        
        obs = (
            wall_sensors + 
            enemy_sensors +
            [hp, x, y, angle, min_enemy_dist, avg_enemy_dist, 
             enemies_nearby, min_wall_dist, avg_wall_dist]
        )
        
        # Pad до 32
        obs = list(obs)
        while len(obs) < self.obs_dim:
            obs.append(0.0)
        
        return np.array(obs[:self.obs_dim], dtype=np.float32)
    
    def think(self, observation):
        obs = self.preprocess_obs(observation)
        
        # Выбор действия
        deterministic = not self.train
        action, shoot, log_prob, value = self.policy.select_action(obs, deterministic)
        
        # Decode action
        dx_raw, dy_raw, angle_delta = action
        
        # Normalize movement
        move = np.array([dx_raw, dy_raw])
        if np.linalg.norm(move) > 0:
            move = move / np.linalg.norm(move)
        
        # Update angle
        self.angle += angle_delta * 0.15
        self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))
        
        command = {
            "dx": float(move[0]),
            "dy": float(move[1]),
            "angle": self.angle,
            "shoot": bool(shoot)
        }
        
        # =========================================
        # LEARNING STEP
        # =========================================
        if self.train and self.prev_obs is not None:
            # Вычисляем reward
            reward = self.reward_shaper.compute_reward(observation, command)
            self.episode_reward += reward
            self.step_count += 1
            
            # Проверяем смерть
            done = observation['self']['hp'] < 0.1
            
            # Сохраняем в буфер
            self.buffer.push(
                self.prev_obs, action, reward, done, log_prob, value
            )
            
            # Обучение
            if len(self.buffer) > 100:
                self.trainer.update(self.buffer, batch_size=32, epochs=2)
            
            # Логирование
            if self.step_count % 100 == 0:
                print(f"📊 Steps: {self.step_count}, "
                      f"Avg Reward: {self.episode_reward/100:.3f}, "
                      f"Buffer: {len(self.buffer)}")
                self.episode_reward = 0
        
        self.prev_obs = obs
        
        # Сброс при смерти
        if observation['self']['hp'] < 0.1:
            self.reward_shaper.reset()
            self.prev_obs = None
        
        return command
    
    def save_model(self, path):
        self.policy.save(path)


# =========================================================
# NETWORK & MAIN
# =========================================================

async def run_bot(model_path=None, train=True, save_interval=500):
    bot = RLBot(model_path=model_path, train=train)
    
    async with websockets.connect(SERVER) as ws:
        await ws.send(json.dumps({
            "type": "hello",
            "name": "RL_SelfLearn",
            "protocol": PROTOCOL_VERSION
        }))
        
        print(f"🤖 RL Bot started | Train: {train} | Model: {model_path or 'random'}")
        
        step_count = 0
        
        while True:
            raw = await ws.recv()
            data = json.loads(raw)
            
            if data.get("type") != "bot_observation_v1":
                continue
            
            command = bot.think(data)
            await ws.send(json.dumps(command))
            
            step_count += 1
            
            # Сохранение модели периодически
            if train and step_count % save_interval == 0:
                bot.save_model(f"rl_bot_checkpoint_{step_count}.pt")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Load model")
    parser.add_argument("--test", action="store_true", help="Test mode (no training)")
    parser.add_argument("--save-interval", type=int, default=500, help="Save every N steps")
    args = parser.parse_args()
    
    train = not args.test
    
    asyncio.run(run_bot(
        model_path=args.model,
        train=train,
        save_interval=args.save_interval
    ))