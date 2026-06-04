# neat_bot_runtime.py
# =========================================================
# REALTIME NEAT BOT
# =========================================================
# FEATURES
# ---------------------------------------------------------
# - ONE websocket forever
# - NO reconnect spam
# - Sequential genome evaluation
# - Recurrent NEAT
# - Continuous controls
# - Reward shaping
# - Frame stacking
# - Persistent checkpoints
# - Stable realtime evolution
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

BOT_NAME = "NEAT_RUNTIME"

PROTOCOL_VERSION = 2

# ---------------------------------------------------------
# OBSERVATION
# ---------------------------------------------------------

STACK_SIZE = 8

FRAME_SIZE = 9

INPUT_SIZE = STACK_SIZE * FRAME_SIZE

# ---------------------------------------------------------
# POPULATION
# ---------------------------------------------------------

POPULATION_SIZE = 32

# ---------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------

ROTATION_SPEED = 0.15

# ---------------------------------------------------------
# EPISODE
# ---------------------------------------------------------

MAX_EPISODE_STEPS = 4000

MAX_IDLE_STEPS = 800

# ---------------------------------------------------------
# REWARDS
# ---------------------------------------------------------

SURVIVAL_REWARD = 0.01

DEATH_PENALTY = -20.0

IDLE_PENALTY = -5.0

WALL_PENALTY = -0.03

ENEMY_VISIBLE_REWARD = 0.02

AIM_REWARD = 0.08

SCORE_REWARD_MULTIPLIER = 50.0

# ---------------------------------------------------------
# RESPAWN
# ---------------------------------------------------------

RESPAWN_HP_THRESHOLD_LOW = 0.3

RESPAWN_HP_THRESHOLD_HIGH = 0.8

RESPAWN_DISTANCE_THRESHOLD = 30.0

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

CONFIG_PATH = "neat_runtime_config.txt"

CHECKPOINT_PREFIX = "neat-runtime-checkpoint-"

BEST_GENOME_FILE = "best_runtime_genome.pkl"

# =========================================================
# NEAT CONFIG
# =========================================================

NEAT_CONFIG = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000000
pop_size              = {POPULATION_SIZE}
reset_on_extinction   = False
no_fitness_termination = False

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
num_inputs              = {INPUT_SIZE}
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

if not os.path.exists(CONFIG_PATH):

    with open(CONFIG_PATH, "w") as f:

        f.write(NEAT_CONFIG)


# =========================================================
# EVOLUTION MANAGER
# =========================================================

class EvolutionManager:

    def __init__(self):

        self.config = neat.Config(

            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,

            CONFIG_PATH
        )

        self.population = neat.Population(
            self.config
        )

        self.population.add_reporter(
            neat.StdOutReporter(True)
        )

        self.stats = neat.StatisticsReporter()

        self.population.add_reporter(
            self.stats
        )

        self.generation = 0

        self.best_fitness = -999999

        self.reset_generation()

    # =====================================================
    # RESET GENERATION
    # =====================================================

    def reset_generation(self):

        self.genomes = list(
            self.population.population.items()
        )

        self.current_index = 0

        self.generation += 1

        print(
            f"\n"
            f"==============================\n"
            f"🧬 GENERATION {self.generation}\n"
            f"=============================="
        )

    # =====================================================
    # CURRENT GENOME
    # =====================================================

    def current_genome(self):

        return self.genomes[
            self.current_index
        ]

    # =====================================================
    # ADVANCE
    # =====================================================

    def advance(self):

        self.current_index += 1

        # -------------------------------------------------
        # GENERATION END
        # -------------------------------------------------

        if self.current_index >= len(self.genomes):

            self.evolve()

            self.reset_generation()

    # =====================================================
    # EVOLVE
    # =====================================================

    def evolve(self):

        # -------------------------------------------------
        # BEST
        # -------------------------------------------------

        best_genome = max(

            self.population.population.values(),

            key=lambda g: g.fitness
            if g.fitness is not None
            else -999999
        )

        best = best_genome.fitness

        avg = np.mean([

            g.fitness if g.fitness is not None else 0

            for g in self.population.population.values()
        ])

        print(
            f"\n"
            f"📊 Generation summary\n"
            f"Best: {best:.1f}\n"
            f"Avg : {avg:.1f}\n"
        )

        # -------------------------------------------------
        # SAVE BEST
        # -------------------------------------------------

        if best > self.best_fitness:

            self.best_fitness = best

            with open(
                BEST_GENOME_FILE,
                "wb"
            ) as f:

                pickle.dump(
                    best_genome,
                    f
                )

            print(
                f"🏆 New best saved "
                f"({best:.1f})"
            )

        # -------------------------------------------------
        # CHECKPOINT
        # -------------------------------------------------

        checkpoint_name = (
            f"{CHECKPOINT_PREFIX}"
            f"{self.generation}"
        )

        with open(
            checkpoint_name,
            "wb"
        ) as f:

            pickle.dump(
                self.population,
                f
            )

        print(
            f"💾 Checkpoint saved: "
            f"{checkpoint_name}"
        )

        # -------------------------------------------------
        # REPRODUCTION
        # -------------------------------------------------

        self.population.population = (
            self.population.reproduction.reproduce(

                self.config,

                self.population.species,

                self.config.pop_size,

                self.generation
            )
        )

        # -------------------------------------------------
        # SPECIATE
        # -------------------------------------------------

        if not self.population.species.species:

            self.population.reporters.complete_extinction()

            self.population.population = (
                self.population.reproduction.create_new(

                    self.config.genome_type,

                    self.config.genome_config,

                    self.config.pop_size
                )
            )

        self.population.species.speciate(

            self.config,

            self.population.population,

            self.generation
        )


# =========================================================
# BOT
# =========================================================

class RuntimeNeatBot:

    def __init__(self):

        self.evolution = EvolutionManager()

        self.genome_id = None

        self.genome = None

        self.net = None

        self.angle = 0.0

        self.history = deque(
            maxlen=STACK_SIZE
        )

        self.load_current_genome()

        self.prev_hp = 1.0

        self.prev_x = 0.0
        self.prev_y = 0.0

        self.prev_score = 0.0

        self.episode_reward = 0.0

        self.steps = 0

        self.idle_steps = 0

    # =====================================================
    # LOAD GENOME
    # =====================================================

    def load_current_genome(self):

        self.genome_id, self.genome = (
            self.evolution.current_genome()
        )

        self.genome.fitness = 0.0

        self.net = neat.nn.RecurrentNetwork.create(

            self.genome,

            self.evolution.config
        )

        self.angle = random.uniform(
            -math.pi,
            math.pi
        )

        self.reset_history()

        self.episode_reward = 0.0

        self.steps = 0

        self.idle_steps = 0

        print(
            f"\n"
            f"🎮 Genome {self.genome_id} started"
        )

    # =====================================================
    # HISTORY
    # =====================================================

    def reset_history(self):

        self.history.clear()

        zero = [0.0] * FRAME_SIZE

        for _ in range(STACK_SIZE):

            self.history.append(
                zero.copy()
            )

    # =====================================================
    # STATE
    # =====================================================

    def build_state(self, data):

        self_data = data["self"]

        wall = data.get(
            "wall_sensors",
            [0.0] * 3
        )

        enemy = data.get(
            "enemy_sensors",
            [0.0] * 3
        )

        frame = [

            *wall,

            *enemy,

            self_data["hp"],

            math.sin(self.angle),

            math.cos(self.angle)
        ]

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

        score = self_data["score"]

        wall = data["wall_sensors"]

        enemy = data["enemy_sensors"]

        reward = SURVIVAL_REWARD

        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        delta_score = (
            score - self.prev_score
        )

        if delta_score > 0:

            self.idle_steps = 0

        else:

            self.idle_steps += 1

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
    # END EPISODE
    # =====================================================

    def end_episode(self):

        self.genome.fitness = (
            self.episode_reward
        )

        print(
            f"☠️ Genome {self.genome_id} finished | "
            f"Fitness: {self.episode_reward:.1f}"
        )

        self.evolution.advance()

        self.load_current_genome()

    # =====================================================
    # MAIN LOOP
    # =====================================================

    async def run(self):

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

                self.episode_reward += reward

                self.steps += 1

                # -------------------------------------------------
                # SAVE PREVIOUS
                # -------------------------------------------------

                self.prev_hp = hp

                self.prev_x = x
                self.prev_y = y

                self.prev_score = self_data["score"]

                # -------------------------------------------------
                # IDLE TIMEOUT
                # -------------------------------------------------

                idle_timeout = (
                    self.idle_steps
                    >=
                    MAX_IDLE_STEPS
                )

                if idle_timeout:

                    self.episode_reward += (
                        IDLE_PENALTY
                    )

                    print(
                        "💤 Idle timeout"
                    )

                # -------------------------------------------------
                # END EPISODE
                # -------------------------------------------------

                if (

                    died

                    or

                    self.steps >= MAX_EPISODE_STEPS

                    or

                    idle_timeout
                ):

                    self.end_episode()


# =========================================================
# MAIN
# =========================================================

async def main():

    bot = RuntimeNeatBot()

    await bot.run()


if __name__ == "__main__":

    asyncio.run(main())