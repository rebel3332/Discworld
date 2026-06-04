# neat_runtime_bot_full.py
# =========================================================
# ADVANCED REALTIME NEAT BOT
# =========================================================
# FEATURES
# ---------------------------------------------------------
# ✔ True persistent NEAT evolution
# ✔ Proper checkpoint restore
# ✔ Sequential evaluation
# ✔ One websocket forever
# ✔ Recurrent neural networks
# ✔ Species preservation
# ✔ Innovation tracking
# ✔ Stable reward shaping
# ✔ Anti-stagnation
# ✔ Reward normalization
# ✔ Elite preservation
# ✔ Automatic checkpoints
# ✔ Best genome saving
# ✔ Runtime-safe evolution
# ✔ Sensor-based learning
# ✔ No reconnects
# ✔ No frame history
# =========================================================

import asyncio
import json
import math
import os
import pickle
import random
import time

import neat
import numpy as np
import websockets


# =========================================================
# CONFIG
# =========================================================

SERVER = "ws://localhost:8000/ws"

BOT_NAME = "NEAT_ADVANCED"

PROTOCOL_VERSION = 2

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

INPUT_SIZE = 9

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

OUTPUT_SIZE = 4

# ---------------------------------------------------------
# POPULATION
# ---------------------------------------------------------

POPULATION_SIZE = 64

# ---------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------

ROTATION_SPEED = 0.15

# ---------------------------------------------------------
# EPISODE
# ---------------------------------------------------------

MAX_EPISODE_STEPS = 6000

MAX_IDLE_STEPS = 1200

# ---------------------------------------------------------
# FITNESS
# ---------------------------------------------------------

KILL_REWARD = 100.0

SURVIVAL_REWARD = 0.01

AIM_REWARD = 0.03

ENEMY_VISIBLE_REWARD = 0.01

MOVE_REWARD = 0.002

WALL_PENALTY = -0.02

DEATH_PENALTY = -15.0

IDLE_PENALTY = -10.0

SPIN_PENALTY = -0.001

REWARD_CLAMP_MIN = -20.0

REWARD_CLAMP_MAX = 200.0

# ---------------------------------------------------------
# RESPAWN
# ---------------------------------------------------------

RESPAWN_HP_THRESHOLD_LOW = 0.3

RESPAWN_HP_THRESHOLD_HIGH = 0.8

RESPAWN_DISTANCE_THRESHOLD = 30.0

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

CONFIG_PATH = "neat_simple_runtime_config.txt"

CHECKPOINT_PREFIX = "neat-simple-checkpoint-"

BEST_GENOME_FILE = "neat_simple_best_genome.pkl"

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
initial_connection    = full_direct

activation_default      = tanh
activation_mutate_rate  = 0.02
activation_options      = tanh sigmoid relu

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0

bias_max_value          = 30.0
bias_min_value          = -30.0

bias_mutate_power       = 0.4
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.3
conn_delete_prob        = 0.1

enabled_default         = True
enabled_mutate_rate     = 0.02

node_add_prob           = 0.15
node_delete_prob        = 0.05

num_hidden              = 0
num_inputs              = {INPUT_SIZE}
num_outputs             = {OUTPUT_SIZE}

response_init_mean      = 1.0
response_init_stdev     = 0.0

response_max_value      = 30.0
response_min_value      = -30.0

response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.5

weight_max_value        = 30
weight_min_value        = -30

weight_mutate_power     = 0.5
weight_mutate_rate      = 0.85
weight_replace_rate     = 0.1

single_structural_mutation = False
structural_mutation_surer = default

[DefaultSpeciesSet]
compatibility_threshold = 3.5

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 40
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
# CHECKPOINT SEARCH
# =========================================================

def find_latest_checkpoint():

    checkpoints = []

    for file in os.listdir("."):

        if file.startswith(CHECKPOINT_PREFIX):

            try:

                generation = int(
                    file.split("-")[-1]
                )

                checkpoints.append(
                    (generation, file)
                )

            except:
                pass

    if not checkpoints:
        return None

    checkpoints.sort()

    return checkpoints[-1][1]


# =========================================================
# EVOLUTION
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

        # -------------------------------------------------
        # RESTORE CHECKPOINT
        # -------------------------------------------------

        latest_checkpoint = (
            find_latest_checkpoint()
        )

        if latest_checkpoint:

            print(
                f"📦 Restoring checkpoint: "
                f"{latest_checkpoint}"
            )

            self.population = (
                neat.Checkpointer.restore_checkpoint(
                    latest_checkpoint
                )
            )

        else:

            print(
                "🌱 Fresh evolution start"
            )

            self.population = neat.Population(
                self.config
            )

        # -------------------------------------------------
        # REPORTERS
        # -------------------------------------------------

        self.population.add_reporter(
            neat.StdOutReporter(True)
        )

        self.stats = neat.StatisticsReporter()

        self.population.add_reporter(
            self.stats
        )

        # -------------------------------------------------
        # CHECKPOINTER
        # -------------------------------------------------

        self.checkpointer = neat.Checkpointer(
            generation_interval=1,
            filename_prefix=CHECKPOINT_PREFIX
        )

        # -------------------------------------------------
        # GENERATION
        # -------------------------------------------------

        self.generation = (
            self.population.generation
        )

        self.best_fitness = -999999

        self.reset_generation()

    # =====================================================
    # RESET GENERATION
    # =====================================================

    def reset_generation(self):

        self.genomes = list(
            self.population.population.items()
        )

        random.shuffle(self.genomes)

        self.current_index = 0

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

        if self.current_index >= len(self.genomes):

            self.evolve()

            self.generation += 1

            self.reset_generation()

    # =====================================================
    # EVOLVE
    # =====================================================

    def evolve(self):

        best_genome = max(

            self.population.population.values(),

            key=lambda g:
                g.fitness
                if g.fitness is not None
                else -999999
        )

        best = best_genome.fitness

        avg = np.mean([

            g.fitness
            if g.fitness is not None
            else 0

            for g in self.population.population.values()
        ])

        print(
            f"\n"
            f"📊 GENERATION SUMMARY\n"
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
                f"🏆 NEW BEST SAVED "
                f"({best:.1f})"
            )

        # -------------------------------------------------
        # REPRODUCTION
        # -------------------------------------------------

        new_population = (
            self.population.reproduction.reproduce(

                self.config,

                self.population.species,

                self.config.pop_size,

                self.generation
            )
        )

        # -------------------------------------------------
        # EXTINCTION
        # -------------------------------------------------

        if not new_population:

            print(
                "☠️ COMPLETE EXTINCTION"
            )

            new_population = (
                self.population.reproduction.create_new(

                    self.config.genome_type,

                    self.config.genome_config,

                    self.config.pop_size
                )
            )

        self.population.population = (
            new_population
        )

        # -------------------------------------------------
        # SPECIATE
        # -------------------------------------------------

        self.population.species.speciate(

            self.config,

            self.population.population,

            self.generation
        )

        # -------------------------------------------------
        # SAVE CHECKPOINT
        # -------------------------------------------------

        self.checkpointer.save_checkpoint(

            self.config,

            self.population.population,

            self.population.species,

            self.generation
        )

        print(
            f"💾 Checkpoint saved "
            f"(generation {self.generation})"
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

        self.prev_hp = 1.0

        self.prev_x = 0.0
        self.prev_y = 0.0

        self.prev_score = 0.0

        self.episode_reward = 0.0

        self.steps = 0

        self.idle_steps = 0

        self.load_current_genome()

    # =====================================================
    # LOAD GENOME
    # =====================================================

    def load_current_genome(self):

        self.genome_id, self.genome = (
            self.evolution.current_genome()
        )

        if self.genome.fitness is None:

            self.genome.fitness = 0.0

        self.net = neat.nn.RecurrentNetwork.create(

            self.genome,

            self.evolution.config
        )

        self.angle = random.uniform(
            -math.pi,
            math.pi
        )

        self.prev_score = 0.0

        self.episode_reward = 0.0

        self.steps = 0

        self.idle_steps = 0

        print(
            f"\n"
            f"🎮 Genome {self.genome_id} started"
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

        state = np.array([

            *wall,

            *enemy,

            self_data["hp"],

            math.sin(self.angle),

            math.cos(self.angle)

        ], dtype=np.float32)

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
        action,
        died
    ):

        reward = 0.0

        self_data = data["self"]

        score = self_data["score"]

        wall = data["wall_sensors"]

        enemy = data["enemy_sensors"]

        # -------------------------------------------------
        # KILL
        # -------------------------------------------------

        delta_score = (
            score - self.prev_score
        )

        if delta_score > 0:

            reward += KILL_REWARD

            self.idle_steps = 0

        else:

            self.idle_steps += 1

        # -------------------------------------------------
        # SURVIVAL
        # -------------------------------------------------

        reward += SURVIVAL_REWARD

        # -------------------------------------------------
        # ENEMY VISIBLE
        # -------------------------------------------------

        reward += (
            max(enemy)
            *
            ENEMY_VISIBLE_REWARD
        )

        # -------------------------------------------------
        # AIM
        # -------------------------------------------------

        reward += (
            enemy[1]
            *
            AIM_REWARD
        )

        # -------------------------------------------------
        # MOVE
        # -------------------------------------------------

        reward += (
            abs(action["move_front_back"])
            *
            MOVE_REWARD
        )

        reward += (
            abs(action["move_left_right"])
            *
            MOVE_REWARD
        )

        # -------------------------------------------------
        # WALL
        # -------------------------------------------------

        if wall[1] > 0.95:

            reward += WALL_PENALTY

        # -------------------------------------------------
        # SPIN
        # -------------------------------------------------

        reward += (
            abs(action["look_delta"])
            *
            SPIN_PENALTY
        )

        # -------------------------------------------------
        # DEATH
        # -------------------------------------------------

        if died:

            reward += DEATH_PENALTY

        # -------------------------------------------------
        # CLAMP
        # -------------------------------------------------

        reward = max(
            REWARD_CLAMP_MIN,
            min(
                reward,
                REWARD_CLAMP_MAX
            )
        )

        return reward

    # =====================================================
    # END EPISODE
    # =====================================================

    def end_episode(self):

        self.genome.fitness = (
            self.episode_reward
        )

        print(
            f"☠️ Genome {self.genome_id} "
            f"finished | "
            f"Fitness: "
            f"{self.episode_reward:.1f}"
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

                outputs = self.net.activate(
                    state
                )

                action = {

                    "move_front_back":
                        float(np.tanh(outputs[0])),

                    "move_left_right":
                        float(np.tanh(outputs[1])),

                    "look_delta":
                        float(np.tanh(outputs[2])),

                    "shoot":
                        1 if outputs[3] > 0 else 0
                }

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

                    action,

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

                self.prev_score = (
                    self_data["score"]
                )

                # -------------------------------------------------
                # IDLE
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
                # END
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