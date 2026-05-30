# qlearning_bot.py
# =========================================================
# SIMPLE TABULAR Q-LEARNING BOT
# =========================================================
# Features:
# - Persistent Q-table
# - Epsilon-greedy exploration
# - Reward shaping
# - Turbo mode friendly
# - Simple discrete state space
# - Relative movement controls
# =========================================================

import asyncio
import json
import math
import os
import pickle
import random
import time

import websockets


# =========================================================
# CONFIG
# =========================================================

SERVER = "ws://localhost:8000/ws"

BOT_NAME = "QLEARN_BOT"

PROTOCOL_VERSION = 2

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

QTABLE_FILE = "qtable.pkl"

# ---------------------------------------------------------
# LEARNING
# ---------------------------------------------------------

LEARNING_RATE = 0.15          # alpha
DISCOUNT = 0.95               # gamma

# ---------------------------------------------------------
# EXPLORATION
# ---------------------------------------------------------

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

# ---------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------

ROTATION_SPEED = 0.15

# ---------------------------------------------------------
# REWARDS
# ---------------------------------------------------------

SURVIVAL_REWARD = 0.1

SCORE_REWARD_MULTIPLIER = 1.0

DEATH_PENALTY = -25.0

# ---------------------------------------------------------
# STATE DISCRETIZATION
# ---------------------------------------------------------

STATE_BUCKETS = 3

# ---------------------------------------------------------
# RESPAWN DETECTION
# ---------------------------------------------------------

RESPAWN_HP_THRESHOLD_LOW = 0.3
RESPAWN_HP_THRESHOLD_HIGH = 0.8

RESPAWN_DISTANCE_THRESHOLD = 30.0


# =========================================================
# ACTION SPACE
# =========================================================

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
        "look_delta": 0.0,
        "shoot": 1
    },

    # 7
    {
        "move_front_back": 0.0,
        "move_left_right": 0.0,
        "look_delta": 0.0,
        "shoot": 1
    },
]


# =========================================================
# Q LEARNER
# =========================================================

class QLearner:

    def __init__(self):

        self.q = {}

        self.epsilon = EPSILON_START

        self.total_updates = 0

        self.load()

    # =====================================================
    # SAVE / LOAD
    # =====================================================

    def save(self):

        data = {

            "q": self.q,

            "epsilon": self.epsilon,

            "updates": self.total_updates
        }

        with open(QTABLE_FILE, "wb") as f:

            pickle.dump(data, f)

        print(
            f"💾 Saved Q-table "
            f"({len(self.q)} states)"
        )

    def load(self):

        if not os.path.exists(QTABLE_FILE):

            print("🌱 Fresh Q-learning start")

            return

        try:

            with open(QTABLE_FILE, "rb") as f:

                data = pickle.load(f)

            self.q = data["q"]

            self.epsilon = data["epsilon"]

            self.total_updates = data["updates"]

            print(
                f"📦 Loaded Q-table "
                f"({len(self.q)} states)"
            )

        except Exception as e:

            print(f"❌ Failed to load Q-table: {e}")

    # =====================================================
    # STATE
    # =====================================================

    def bucket(self, v):

        v = max(0.0, min(1.0, v))

        if v < 0.33:
            return 0

        if v < 0.66:
            return 1

        return 2

    def observation_to_state(self, obs):

        wall = obs.get(
            "wall_sensors",
            [0.0] * 3
        )

        enemy = obs.get(
            "enemy_sensors",
            [0.0] * 3
        )

        state = tuple(

            [self.bucket(v) for v in wall]

            +

            [self.bucket(v) for v in enemy]
        )

        return state

    # =====================================================
    # Q TABLE
    # =====================================================

    def ensure_state(self, state):

        if state not in self.q:

            self.q[state] = [

                0.0

                for _ in range(len(ACTIONS))
            ]

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def choose_action(self, state):

        self.ensure_state(state)

        # exploration
        if random.random() < self.epsilon:

            return random.randint(
                0,
                len(ACTIONS) - 1
            )

        # exploitation
        qvals = self.q[state]

        return max(
            range(len(qvals)),
            key=lambda i: qvals[i]
        )

    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        self.ensure_state(state)

        self.ensure_state(next_state)

        old_q = self.q[state][action]

        next_max = max(
            self.q[next_state]
        )

        new_q = old_q + LEARNING_RATE * (

            reward

            +

            DISCOUNT * next_max

            -

            old_q
        )

        self.q[state][action] = new_q

        self.total_updates += 1

        # epsilon decay
        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY
        )

        # autosave
        if self.total_updates % 1000 == 0:

            self.save()


# =========================================================
# BOT
# =========================================================

class QLearningBot:

    def __init__(self):

        self.agent = QLearner()

        self.angle = random.uniform(
            -math.pi,
            math.pi
        )

        self.prev_hp = 1.0

        self.prev_x = 0.0
        self.prev_y = 0.0

        self.prev_score = 0.0

        self.prev_state = None
        self.prev_action = None

        self.episode_reward = 0.0

        self.episode_steps = 0

        self.episode = 1

    # =====================================================
    # RESPAWN DETECTION
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
        score,
        died
    ):

        reward = SURVIVAL_REWARD

        # score delta reward
        score_delta = (
            score - self.prev_score
        )

        reward += (
            score_delta
            *
            SCORE_REWARD_MULTIPLIER
        )

        if died:

            reward += DEATH_PENALTY

        return reward

    # =====================================================
    # MAIN LOOP
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

                score = self_data.get(
                    "score",
                    0
                )

                # -----------------------------------------
                # STATE
                # -----------------------------------------

                state = self.agent.observation_to_state(
                    data
                )

                # -----------------------------------------
                # RESPAWN
                # -----------------------------------------

                died = self.detect_respawn(
                    hp,
                    x,
                    y
                )

                # -----------------------------------------
                # REWARD
                # -----------------------------------------

                reward = self.compute_reward(
                    score,
                    died
                )

                self.episode_reward += reward

                # -----------------------------------------
                # LEARN
                # -----------------------------------------

                if (
                    self.prev_state is not None
                    and
                    self.prev_action is not None
                ):

                    self.agent.update(

                        self.prev_state,

                        self.prev_action,

                        reward,

                        state
                    )

                # -----------------------------------------
                # ACTION
                # -----------------------------------------

                action_idx = self.agent.choose_action(
                    state
                )

                action = ACTIONS[action_idx]

                # -----------------------------------------
                # LOOK
                # -----------------------------------------

                self.angle += (

                    action["look_delta"]

                    *

                    ROTATION_SPEED
                )

                self.angle = math.atan2(
                    math.sin(self.angle),
                    math.cos(self.angle)
                )

                # -----------------------------------------
                # SEND
                # -----------------------------------------

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

                # -----------------------------------------
                # EPISODE END
                # -----------------------------------------

                if died:

                    print(

                        f"📊 Episode {self.episode:5d} | "

                        f"Reward: {self.episode_reward:8.1f} | "

                        f"Epsilon: {self.agent.epsilon:.3f} | "

                        f"States: {len(self.agent.q)}"
                    )

                    self.episode += 1

                    self.episode_reward = 0.0

                # -----------------------------------------
                # SAVE PREVIOUS
                # -----------------------------------------

                self.prev_hp = hp

                self.prev_x = x
                self.prev_y = y

                self.prev_score = score

                self.prev_state = state

                self.prev_action = action_idx

                self.episode_steps += 1


# =========================================================
# MAIN
# =========================================================

async def main():

    bot = QLearningBot()

    await bot.run_forever()


if __name__ == "__main__":

    asyncio.run(main())