# neat_bot.py
# =========================================================
# NEAT BOT
# =========================================================
# Features:
# - Recurrent NEAT
# - Continuous Controls
# - Persistent Checkpoints
# - Turbo Mode Ready
# - Parallel-Friendly
# - Reward Shaping
# - Frame History
# - Species Evolution
# =========================================================

import asyncio
import json
import math
import os
import pickle
import random
from collections import deque

import neat
import numpy as np
import websockets


# =========================================================
# CONFIG
# =========================================================

SERVER = "ws://localhost:8000/ws"

BOT_NAME = "NEAT_BOT"

PROTOCOL_VERSION = 2

# ---------------------------------------------------------
# OBSERVATION
# ---------------------------------------------------------

STACK_SIZE = 8

FRAME_SIZE = 9

INPUT_SIZE = STACK_SIZE * FRAME_SIZE

# ---------------------------------------------------------
# ACTIONS
# ---------------------------------------------------------

ROTATION_SPEED = 0.15

# ---------------------------------------------------------
# FITNESS
# ---------------------------------------------------------

SURVIVAL_REWARD = 0.01

DEATH_PENALTY = -25.0

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

# ---------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------

CHECKPOINT_PREFIX = "neat-checkpoint-"

BEST_GENOME_FILE = "best_neat_genome.pkl"

# ---------------------------------------------------------
# GENERATIONS
# ---------------------------------------------------------

MAX_EPISODE_STEPS = 4000

# =========================================================
# NEAT CONFIG FILE
# =========================================================

NEAT_CONFIG = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 500000
pop_size              = 64
reset_on_extinction   = False

[DefaultGenome]
feed_forward          = False
initial_connection    = full

activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0

bias_max_value          = 30.0
bias_min_value          = -30.0

bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.5
conn_delete_prob        = 0.3

enabled_default         = True
enabled_mutate_rate     = 0.01

node_add_prob           = 0.2
node_delete_prob        = 0.1

num_hidden              = 0
num_inputs              = 72
num_outputs             = 4

response_init_mean      = 1.0
response_init_stdev     = 0.0

response_max_value      = 30.0
response_min_value      = -30.0

response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0

weight_max_value        = 30
weight_min_value        = -30

weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 30
species_elitism      = 3

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.2
"""


# =========================================================
# SAVE CONFIG
# =========================================================

CONFIG_PATH = "neat_config.txt"

if not os.path.exists(CONFIG_PATH):

    with open(CONFIG_PATH, "w") as f:

        f.write(NEAT_CONFIG)


# =========================================================
# BOT
# =========================================================

class NeatBot:

    def __init__(self, genome, config):

        self.genome = genome

        self.net = neat.nn.RecurrentNetwork.create(
            genome,
            config
        )

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

        self.total_reward = 0.0

        self.steps = 0

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

        return [

            *wall,

            *enemy,

            hp,

            math.sin(self.angle),

            math.cos(self.angle)
        ]

    # =====================================================
    # STATE
    # =====================================================

    def build_state(self, data):

        frame = self.build_frame(data)

        self.history.append(frame)

        return np.array(
            self.history,
            dtype=np.float32
        ).flatten()

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

        wall = data["wall_sensors"]

        enemy = data["enemy_sensors"]

        score = self_data["score"]

        reward = SURVIVAL_REWARD

        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        delta_score = (
            score - self.prev_score
        )

        reward += (
            delta_score
            *
            SCORE_REWARD_MULTIPLIER
        )

        # -------------------------------------------------
        # ENEMY
        # -------------------------------------------------

        reward += (
            max(enemy)
            *
            ENEMY_VISIBLE_REWARD
        )

        reward += (
            enemy[1]
            *
            AIM_REWARD
        )

        # -------------------------------------------------
        # WALL
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
    # RUN
    # =====================================================

    async def run(self):

        async with websockets.connect(SERVER) as ws:

            await ws.send(json.dumps({

                "type": "hello",

                "name": BOT_NAME,

                "protocol":
                    PROTOCOL_VERSION

            }))

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
                # NETWORK
                # -------------------------------------------------

                outputs = self.net.activate(state)

                move_front_back = float(
                    np.tanh(outputs[0])
                )

                move_left_right = float(
                    np.tanh(outputs[1])
                )

                look_delta = float(
                    np.tanh(outputs[2])
                )

                shoot = 1 if outputs[3] > 0 else 0

                # -------------------------------------------------
                # LOOK
                # -------------------------------------------------

                self.angle += (
                    look_delta
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
                        move_front_back,

                    "move_left_right":
                        move_left_right,

                    "angle":
                        self.angle,

                    "shoot":
                        shoot

                }))

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

                self.total_reward += reward

                self.steps += 1

                # -------------------------------------------------
                # SAVE PREVIOUS
                # -------------------------------------------------

                self.prev_hp = hp

                self.prev_x = x
                self.prev_y = y

                self.prev_score = self_data["score"]

                # -------------------------------------------------
                # DONE
                # -------------------------------------------------

                if died or self.steps >= MAX_EPISODE_STEPS:

                    return self.total_reward


# =========================================================
# EVAL
# =========================================================

async def eval_genome(genome, config):

    genome.fitness = 0.0

    try:

        bot = NeatBot(genome, config)

        fitness = await bot.run()

        genome.fitness = fitness

    except Exception as e:

        print(f"❌ Genome failed: {e}")

        genome.fitness = -100.0

    return genome.fitness


# =========================================================
# EVAL ALL
# =========================================================

def eval_genomes(genomes, config):

    async def run_all():

        # # Спавн сразу всех игроков ассинхронно
        # tasks = []
        # for genome_id, genome in genomes:
        #     tasks.append(
        #         eval_genome(genome, config)
        #     )
        # results = await asyncio.gather(*tasks)

        # Спавн ботов по одному
        results = []
        for genome_id, genome in genomes:
            fitness = await eval_genome(
                genome,
                config
            )
            results.append(fitness)



        best = max(results)

        avg = sum(results) / len(results)

        print(
            f"📊 "
            f"Best: {best:8.1f} | "
            f"Avg: {avg:8.1f}"
        )

    asyncio.run(run_all())


# =========================================================
# MAIN
# =========================================================

def run_neat():

    config = neat.Config(

        neat.DefaultGenome,

        neat.DefaultReproduction,

        neat.DefaultSpeciesSet,

        neat.DefaultStagnation,

        CONFIG_PATH
    )

    # -----------------------------------------------------
    # RESTORE
    # -----------------------------------------------------

    checkpoints = [

        f for f in os.listdir(".")

        if f.startswith(CHECKPOINT_PREFIX)
    ]

    if checkpoints:

        latest = sorted(checkpoints)[-1]

        print(f"📦 Restoring {latest}")

        population = neat.Checkpointer.restore_checkpoint(
            latest
        )

    else:

        print("🌱 Fresh NEAT start")

        population = neat.Population(config)

    # -----------------------------------------------------
    # REPORTERS
    # -----------------------------------------------------

    population.add_reporter(
        neat.StdOutReporter(True)
    )

    stats = neat.StatisticsReporter()

    population.add_reporter(stats)

    population.add_reporter(

        neat.Checkpointer(

            generation_interval=5,

            filename_prefix=CHECKPOINT_PREFIX
        )
    )

    # -----------------------------------------------------
    # RUN
    # -----------------------------------------------------

    winner = population.run(
        eval_genomes
    )

    # -----------------------------------------------------
    # SAVE BEST
    # -----------------------------------------------------

    with open(BEST_GENOME_FILE, "wb") as f:

        pickle.dump(winner, f)

    print(
        f"🏆 Best genome saved "
        f"to {BEST_GENOME_FILE}"
    )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    run_neat()