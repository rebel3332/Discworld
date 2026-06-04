# dqn_bot.py
# =========================================================
# DOUBLE DQN BOT
# =========================================================
# Features:
# - Double DQN
# - Replay Buffer
# - Frame Stacking
# - Persistent Checkpoints
# - GPU Support
# - Reward Shaping
# - Target Network
# - Turbo Mode Ready
# - Stable Training
# =========================================================

import asyncio
import json
import math
import os
import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import websockets


# =========================================================
# CONFIG
# =========================================================

SERVER = "ws://localhost:8000/ws"

BOT_NAME = "DQN_BOT"

PROTOCOL_VERSION = 2

# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------
# OBSERVATION
# ---------------------------------------------------------

STACK_SIZE = 8

FRAME_SIZE = 9

STATE_SIZE = STACK_SIZE * FRAME_SIZE

# ---------------------------------------------------------
# ACTIONS
# ---------------------------------------------------------

ROTATION_SPEED = 0.15

ACTIONS = [

    # 0
    {
        "move_front_back": 1.0,
        "move_left_right": 0.0,
        "look_delta": 0.0,
        "shoot": 0
    },

    # 1
    {
        "move_front_back": -1.0,
        "move_left_right": 0.0,
        "look_delta": 0.0,
        "shoot": 0
    },

    # 2
    {
        "move_front_back": 0.0,
        "move_left_right": -1.0,
        "look_delta": 0.0,
        "shoot": 0
    },

    # 3
    {
        "move_front_back": 0.0,
        "move_left_right": 1.0,
        "look_delta": 0.0,
        "shoot": 0
    },

    # 4
    {
        "move_front_back": 0.0,
        "move_left_right": 0.0,
        "look_delta": -1.0,
        "shoot": 0
    },

    # 5
    {
        "move_front_back": 0.0,
        "move_left_right": 0.0,
        "look_delta": 1.0,
        "shoot": 0
    },

    # 6
    {
        "move_front_back": 1.0,
        "move_left_right": 0.0,
        "look_delta": -0.3,
        "shoot": 1
    },

    # 7
    {
        "move_front_back": 1.0,
        "move_left_right": 0.0,
        "look_delta": 0.3,
        "shoot": 1
    },

    # 8
    {
        "move_front_back": 0.0,
        "move_left_right": -1.0,
        "look_delta": 0.0,
        "shoot": 1
    },

    # 9
    {
        "move_front_back": 0.0,
        "move_left_right": 1.0,
        "look_delta": 0.0,
        "shoot": 1
    },

    # 10
    {
        "move_front_back": 0.0,
        "move_left_right": 0.0,
        "look_delta": 0.0,
        "shoot": 1
    },
]

ACTION_COUNT = len(ACTIONS)

# ---------------------------------------------------------
# DQN
# ---------------------------------------------------------

GAMMA = 0.99

LEARNING_RATE = 0.00025

BATCH_SIZE = 64

TARGET_UPDATE_INTERVAL = 2000

TRAIN_INTERVAL = 4

# ---------------------------------------------------------
# REPLAY BUFFER
# ---------------------------------------------------------

REPLAY_SIZE = 100_000

MIN_REPLAY_TO_TRAIN = 5000

# ---------------------------------------------------------
# EPSILON
# ---------------------------------------------------------

EPSILON_START = 1.0

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.999995

# ---------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------

CHECKPOINT_FILE = "dqn_checkpoint.pt"

# ---------------------------------------------------------
# REWARDS
# ---------------------------------------------------------

SURVIVAL_REWARD = 0.01

DEATH_PENALTY = -20.0

WALL_PENALTY = -0.03

ENEMY_VISIBLE_REWARD = 0.02

AIM_REWARD = 0.08

SCORE_REWARD_MULTIPLIER = 50.0

# ---------------------------------------------------------
# RESPAWN DETECTION
# ---------------------------------------------------------

RESPAWN_HP_THRESHOLD_LOW = 0.3

RESPAWN_HP_THRESHOLD_HIGH = 0.8

RESPAWN_DISTANCE_THRESHOLD = 30.0


# =========================================================
# NETWORK
# =========================================================

class DQN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, ACTION_COUNT)
        )

    def forward(self, x):

        return self.net(x)


# =========================================================
# REPLAY BUFFER
# =========================================================

class ReplayBuffer:

    def __init__(self, capacity):

        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append((
            state,
            action,
            reward,
            next_state,
            done
        ))

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        return (

            np.array(states, dtype=np.float32),

            np.array(actions),

            np.array(rewards, dtype=np.float32),

            np.array(next_states, dtype=np.float32),

            np.array(dones, dtype=np.float32)
        )

    def __len__(self):

        return len(self.buffer)


# =========================================================
# AGENT
# =========================================================

class DQNAgent:

    def __init__(self):

        self.online_net = DQN().to(DEVICE)

        self.target_net = DQN().to(DEVICE)

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=LEARNING_RATE
        )

        self.replay = ReplayBuffer(REPLAY_SIZE)

        self.epsilon = EPSILON_START

        self.steps = 0

        self.episodes = 0

        self.load()

    # =====================================================
    # SAVE / LOAD
    # =====================================================

    def save(self):

        torch.save({

            "online":
                self.online_net.state_dict(),

            "target":
                self.target_net.state_dict(),

            "optimizer":
                self.optimizer.state_dict(),

            "epsilon":
                self.epsilon,

            "steps":
                self.steps,

            "episodes":
                self.episodes

        }, CHECKPOINT_FILE)

        print("💾 Checkpoint saved")

    def load(self):

        if not os.path.exists(CHECKPOINT_FILE):

            print("🌱 Fresh DQN start")

            return

        checkpoint = torch.load(
            CHECKPOINT_FILE,
            map_location=DEVICE
        )

        self.online_net.load_state_dict(
            checkpoint["online"]
        )

        self.target_net.load_state_dict(
            checkpoint["target"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]

        self.steps = checkpoint["steps"]

        self.episodes = checkpoint["episodes"]

        print(
            f"📦 Loaded checkpoint | "
            f"episodes={self.episodes} "
            f"epsilon={self.epsilon:.3f}"
        )

    # =====================================================
    # ACTION
    # =====================================================

    def select_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(
                0,
                ACTION_COUNT - 1
            )

        with torch.no_grad():

            state_t = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            qvals = self.online_net(state_t)

            return int(torch.argmax(qvals).item())

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        if len(self.replay) < MIN_REPLAY_TO_TRAIN:

            return

        states, actions, rewards, next_states, dones = (
            self.replay.sample(BATCH_SIZE)
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=DEVICE
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=DEVICE
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=DEVICE
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=DEVICE
        )

        # -------------------------------------------------
        # Q(s,a)
        # -------------------------------------------------

        current_q = self.online_net(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # -------------------------------------------------
        # Double DQN
        # -------------------------------------------------

        with torch.no_grad():

            next_actions = torch.argmax(
                self.online_net(next_states),
                dim=1
            )

            next_q = self.target_net(next_states)

            next_q = next_q.gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q = rewards + (
                (1 - dones)
                *
                GAMMA
                *
                next_q
            )

        # -------------------------------------------------
        # LOSS
        # -------------------------------------------------

        loss = nn.functional.smooth_l1_loss(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            10.0
        )

        self.optimizer.step()

        # -------------------------------------------------
        # TARGET UPDATE
        # -------------------------------------------------

        if self.steps % TARGET_UPDATE_INTERVAL == 0:

            self.target_net.load_state_dict(
                self.online_net.state_dict()
            )

            print("🔄 Target network updated")

        # -------------------------------------------------
        # EPSILON
        # -------------------------------------------------

        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY
        )


# =========================================================
# BOT
# =========================================================

class DQNBot:

    def __init__(self):

        self.agent = DQNAgent()

        self.angle = random.uniform(
            -math.pi,
            math.pi
        )

        self.history = deque(
            maxlen=STACK_SIZE
        )

        self.reset_history()

        self.prev_hp = 1.0

        self.prev_x = 0.0
        self.prev_y = 0.0

        self.prev_score = 0.0

        self.prev_state = None

        self.prev_action = None

        self.episode_reward = 0.0

    # =====================================================
    # HISTORY
    # =====================================================

    def reset_history(self):

        self.history.clear()

        zero = [0.0] * FRAME_SIZE

        for _ in range(STACK_SIZE):

            self.history.append(zero.copy())

    # =====================================================
    # FRAME
    # =====================================================

    def build_frame(self, data):

        self_data = data["self"]

        wall = data.get(
            "wall_sensors",
            [0.0] * 3
        )

        enemy = data.get(
            "enemy_sensors",
            [0.0] * 3
        )

        hp = self_data["hp"]

        angle = self.angle

        return [

            *wall,

            *enemy,

            hp,

            math.sin(angle),

            math.cos(angle)
        ]

    # =====================================================
    # STATE
    # =====================================================

    def build_state(self, data):

        frame = self.build_frame(data)

        self.history.append(frame)

        state = np.array(
            self.history,
            dtype=np.float32
        ).flatten()

        return state

    # =====================================================
    # RESPAWN
    # =====================================================

    def detect_respawn(
        self,
        hp,
        x,
        y
    ):

        hp_respawn = (

            self.prev_hp
            <
            RESPAWN_HP_THRESHOLD_LOW

            and

            hp
            >
            RESPAWN_HP_THRESHOLD_HIGH
        )

        dist = math.hypot(
            x - self.prev_x,
            y - self.prev_y
        )

        teleport_respawn = (
            dist >
            RESPAWN_DISTANCE_THRESHOLD
        )

        return (
            hp_respawn
            or
            teleport_respawn
        )

    # =====================================================
    # REWARD
    # =====================================================

    def compute_reward(
        self,
        data,
        died
    ):

        self_data = data["self"]

        score = self_data["score"]

        wall = data["wall_sensors"]

        enemy = data["enemy_sensors"]

        reward = SURVIVAL_REWARD

        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        score_delta = (
            score - self.prev_score
        )

        reward += (
            score_delta
            *
            SCORE_REWARD_MULTIPLIER
        )

        # -------------------------------------------------
        # ENEMY VISIBLE
        # -------------------------------------------------

        enemy_visible = max(enemy)

        reward += (
            enemy_visible
            *
            ENEMY_VISIBLE_REWARD
        )

        # -------------------------------------------------
        # AIM BONUS
        # -------------------------------------------------

        reward += (
            enemy[1]
            *
            AIM_REWARD
        )

        # -------------------------------------------------
        # WALL PENALTY
        # -------------------------------------------------

        if wall[1] > 0.9:

            reward += WALL_PENALTY

        # -------------------------------------------------
        # DEATH
        # -------------------------------------------------

        if died:

            reward += DEATH_PENALTY

        return reward

    # =====================================================
    # LOOP
    # =====================================================

    async def run_forever(self):

        async with websockets.connect(SERVER) as ws:

            await ws.send(json.dumps({

                "type": "hello",

                "name": BOT_NAME,

                "protocol":
                    PROTOCOL_VERSION

            }))

            print("🟢 Connected")

            while True:

                raw = await ws.recv()

                data = json.loads(raw)

                if (
                    data.get("type")
                    !=
                    "bot_observation_v1"
                ):
                    continue

                self_data = data["self"]

                hp = self_data["hp"]

                x = self_data["x"]

                y = self_data["y"]

                # -------------------------------------------------
                # STATE
                # -------------------------------------------------

                state = self.build_state(data)

                # -------------------------------------------------
                # RESPAWN
                # -------------------------------------------------

                died = self.detect_respawn(
                    hp,
                    x,
                    y
                )

                # -------------------------------------------------
                # REWARD
                # -------------------------------------------------

                reward = self.compute_reward(
                    data,
                    died
                )

                self.episode_reward += reward

                # -------------------------------------------------
                # LEARN
                # -------------------------------------------------

                if (
                    self.prev_state is not None
                    and
                    self.prev_action is not None
                ):

                    self.agent.replay.push(

                        self.prev_state,

                        self.prev_action,

                        reward,

                        state,

                        died
                    )

                    if self.agent.steps % TRAIN_INTERVAL == 0:

                        self.agent.train()

                # -------------------------------------------------
                # ACTION
                # -------------------------------------------------

                action_idx = self.agent.select_action(
                    state
                )

                action = ACTIONS[action_idx]

                # -------------------------------------------------
                # LOOK
                # -------------------------------------------------

                self.angle += (

                    action["look_delta"]

                    *

                    ROTATION_SPEED
                )

                self.angle = math.atan2(
                    math.sin(self.angle),
                    math.cos(self.angle)
                )

                # -------------------------------------------------
                # SEND
                # -------------------------------------------------

                await ws.send(json.dumps({

                    "move_front_back":
                        action["move_front_back"],

                    "move_left_right":
                        action["move_left_right"],

                    "angle":
                        self.angle,

                    "shoot":
                        action["shoot"]

                }))

                # -------------------------------------------------
                # SAVE PREVIOUS
                # -------------------------------------------------

                self.prev_hp = hp

                self.prev_x = x
                self.prev_y = y

                self.prev_score = self_data["score"]

                self.prev_state = state

                self.prev_action = action_idx

                self.agent.steps += 1

                # -------------------------------------------------
                # EPISODE END
                # -------------------------------------------------

                if died:

                    self.agent.episodes += 1

                    print(

                        f"📊 Episode "
                        f"{self.agent.episodes:6d} | "

                        f"Reward: "
                        f"{self.episode_reward:8.1f} | "

                        f"Epsilon: "
                        f"{self.agent.epsilon:.3f} | "

                        f"Replay: "
                        f"{len(self.agent.replay)}"
                    )

                    self.reset_history()

                    self.episode_reward = 0.0

                    if self.agent.episodes % 25 == 0:

                        self.agent.save()


# =========================================================
# MAIN
# =========================================================

async def main():

    print(f"🔥 Using device: {DEVICE}")

    bot = DQNBot()

    await bot.run_forever()


if __name__ == "__main__":

    asyncio.run(main())