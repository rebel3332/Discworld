# bot_ppo.py

import asyncio
import json
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import websockets


"""
PPO BOT for game
======================

Онлайн PPO бот.
Обучается прямо во время игры.

Особенности:
- shared neural brain
- online learning
- PPO policy
- realtime websocket gameplay
- self-learning

pip install torch numpy websockets
"""


SERVER = "ws://127.0.0.1:8000/ws"

MODEL_PATH = "ppo_shared.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_SIZE = 12
ACTION_SIZE = 4


# =========================================================
# PPO NETWORK
# =========================================================

class PPOModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        shared = self.shared(x)

        logits = self.policy(shared)

        value = self.value(shared)

        return logits, value


# =========================================================
# MEMORY
# =========================================================

class Memory:

    def __init__(self):

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


# =========================================================
# BOT
# =========================================================

class PPOBot:

    def __init__(self):

        self.my_id = None

        self.model = PPOModel().to(DEVICE)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0003
        )

        self.memory = Memory()

        self.gamma = 0.99
        self.clip_epsilon = 0.2

        self.prev_hp = 100
        self.prev_score = 0

        self.ticks = 0

        self.load_model()

    # =====================================================
    # SAVE / LOAD
    # =====================================================

    def save_model(self):

        torch.save(
            self.model.state_dict(),
            MODEL_PATH
        )

    def load_model(self):

        try:

            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=DEVICE)
            )

            print("MODEL LOADED")

        except:
            print("NEW MODEL CREATED")

    # =====================================================
    # MAIN LOOP
    # =====================================================

    async def run(self):

        async with websockets.connect(SERVER) as ws:

            print("PPO BOT CONNECTED")

            await ws.send(json.dumps({
                "type": "hello",
                "name": "PPOBot"
            }))

            while True:

                try:

                    msg = await ws.recv()

                    data = json.loads(msg)

                    if data.get("type") == "welcome":

                        self.my_id = data["player_id"]

                        print("MY ID:", self.my_id)

                        continue

                    await self.process_state(ws, data)

                except Exception as e:

                    print("BOT ERROR:", e)

                    break

    # =====================================================
    # PROCESS
    # =====================================================

    async def process_state(self, ws, state):

        players = state.get("players", [])
        enemies = state.get("enemies", [])
        bullets = state.get("bullets", [])

        me = None

        for p in players:
            if p["id"] == self.my_id:
                me = p
                break

        if not me:
            return

        obs = self.build_state(
            me,
            players,
            enemies,
            bullets
        )

        action, log_prob, value = self.select_action(obs)

        dx, dy, shoot, angle = self.decode_action(
            action,
            me,
            players,
            enemies
        )

        reward = self.calculate_reward(me)

        self.memory.states.append(obs)
        self.memory.actions.append(action)
        self.memory.log_probs.append(log_prob)
        self.memory.values.append(value)
        self.memory.rewards.append(reward)
        self.memory.dones.append(False)

        await ws.send(json.dumps({
            "dx": dx,
            "dy": dy,
            "shoot": shoot,
            "angle": angle
        }))

        self.prev_hp = me["hp"]
        self.prev_score = me["score"]

        self.ticks += 1

        # train periodically
        if len(self.memory.states) >= 2048:

            self.train()

            self.memory.clear()

            self.save_model()

            print("MODEL UPDATED")

        if self.ticks % 300 == 0:

            print(
                "ticks:",
                self.ticks,
                "memory:",
                len(self.memory.states)
            )

    # =====================================================
    # STATE ENCODER
    # =====================================================

    def build_state(self, me, players, enemies, bullets):

        nearest_enemy_dx = 0
        nearest_enemy_dy = 0

        nearest_player_dx = 0
        nearest_player_dy = 0

        nearest_bullet_dx = 0
        nearest_bullet_dy = 0

        # nearest enemy
        best_dist = 999999

        for e in enemies:

            dx = e["x"] - me["x"]
            dy = e["y"] - me["y"]

            dist = math.hypot(dx, dy)

            if dist < best_dist:

                best_dist = dist

                nearest_enemy_dx = dx / 800
                nearest_enemy_dy = dy / 600

        # nearest player
        best_dist = 999999

        for p in players:

            if p["id"] == self.my_id:
                continue

            dx = p["x"] - me["x"]
            dy = p["y"] - me["y"]

            dist = math.hypot(dx, dy)

            if dist < best_dist:

                best_dist = dist

                nearest_player_dx = dx / 800
                nearest_player_dy = dy / 600

        # nearest bullet
        best_dist = 999999

        for b in bullets:

            dx = b["x"] - me["x"]
            dy = b["y"] - me["y"]

            dist = math.hypot(dx, dy)

            if dist < best_dist:

                best_dist = dist

                nearest_bullet_dx = dx / 800
                nearest_bullet_dy = dy / 600

        return np.array([

            me["hp"] / 100,

            me["x"] / 800,
            me["y"] / 600,

            nearest_enemy_dx,
            nearest_enemy_dy,

            nearest_player_dx,
            nearest_player_dy,

            nearest_bullet_dx,
            nearest_bullet_dy,

            len(enemies) / 20,
            len(players) / 10,
            len(bullets) / 30

        ], dtype=np.float32)

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def select_action(self, obs):

        state = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

        logits, value = self.model(state)

        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action).item(),
            value.item()
        )

    # =====================================================
    # ACTION DECODER
    # =====================================================

    def decode_action(self, action, me, players, enemies):

        targets = []

        targets.extend(enemies)

        for p in players:
            if p["id"] != self.my_id:
                targets.append(p)

        if not targets:

            return 0, 0, False, 0

        target = min(
            targets,
            key=lambda t: math.hypot(
                t["x"] - me["x"],
                t["y"] - me["y"]
            )
        )

        dx = target["x"] - me["x"]
        dy = target["y"] - me["y"]

        dist = math.hypot(dx, dy)

        if dist <= 0:

            return 0, 0, False, 0

        nx = dx / dist
        ny = dy / dist

        angle = math.atan2(dy, dx)

        # 0 = attack
        if action == 0:

            return nx, ny, True, angle

        # 1 = retreat
        elif action == 1:

            return -nx, -ny, True, angle

        # 2 = strafe left
        elif action == 2:

            return -ny, nx, True, angle

        # 3 = strafe right
        elif action == 3:

            return ny, -nx, True, angle

        return 0, 0, False, angle

    # =====================================================
    # REWARD
    # =====================================================

    def calculate_reward(self, me):

        reward = 0

        score_gain = me["score"] - self.prev_score
        hp_loss = self.prev_hp - me["hp"]

        reward += score_gain * 1.5

        reward -= hp_loss * 2.0

        reward += 0.1
        # print(f"Reward: {reward:.2f} (Score Gain: {score_gain}, HP Loss: {hp_loss})")
        normalized_reward = reward / 31.1
        return normalized_reward

    # =====================================================
    # PPO TRAIN
    # =====================================================

    def train(self):

        states = torch.FloatTensor(
            np.array(self.memory.states)
        ).to(DEVICE)

        actions = torch.LongTensor(
            self.memory.actions
        ).to(DEVICE)

        old_log_probs = torch.FloatTensor(
            self.memory.log_probs
        ).to(DEVICE)

        rewards = self.compute_returns(
            self.memory.rewards
        )

        rewards = torch.FloatTensor(rewards).to(DEVICE)

        logits, values = self.model(states)

        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)

        new_log_probs = dist.log_prob(actions)

        advantages = rewards - values.squeeze()
        print(f"Advantages: {advantages}")
        # advantage normalization
        # advantages = (
        #     advantages - advantages.mean()
        # ) / (advantages.std() + 1e-8)
        # print(f"Normalized Advantages: {advantages}")

        # Ниже пример вывода, видно что модель не учится
        # Advantages: tensor([-4.0867, -4.1250, 16.0347,  ..., -0.0698, -0.0695, -0.0692], grad_fn=<SubBackward0>)
        # Normalized Advantages: tensor([-1.0767, -1.0772, -0.8055,  ..., -1.0225, -1.0225, -1.0225], grad_fn=<DivBackward0>)
        # LOSS: 0.4998 ACTOR: -0.0 CRITIC: 0.9995

        ratio = torch.exp(
            new_log_probs - old_log_probs
        )

        surr1 = ratio * advantages

        surr2 = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon
        ) * advantages

        actor_loss = -torch.min(
            surr1,
            surr2
        ).mean()

        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        print(
            "LOSS:",
            round(loss.item(), 4),
            "ACTOR:",
            round(actor_loss.item(), 4),
            "CRITIC:",
            round(critic_loss.item(), 4)
        )

    # =====================================================
    # RETURNS
    # =====================================================

    def compute_returns(self, rewards):

        returns = []

        discounted = 0

        for reward in reversed(rewards):

            discounted = reward + self.gamma * discounted

            returns.insert(0, discounted)

        return returns


# =========================================================
# START
# =========================================================

async def main():

    bot = PPOBot()

    await bot.run()


if __name__ == "__main__":

    asyncio.run(main())