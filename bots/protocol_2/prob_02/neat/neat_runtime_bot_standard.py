# neat_runtime_bot_standard.py

import asyncio
import json
import math
import os
import pickle
import random

import neat
import numpy as np
import websockets


# =========================================================
# CONFIG
# =========================================================

SERVER = "ws://localhost:8000/ws"

BOT_NAME = "NEAT_STANDARD"

PROTOCOL_VERSION = 2

INPUT_SIZE = 9
OUTPUT_SIZE = 4

POPULATION_SIZE = 64

ROTATION_SPEED = 0.15

MAX_EPISODE_STEPS = 6000
MAX_IDLE_STEPS = 1200

# =========================================================
# FITNESS
# =========================================================

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

RESPAWN_HP_THRESHOLD_LOW = 0.3
RESPAWN_HP_THRESHOLD_HIGH = 0.8
RESPAWN_DISTANCE_THRESHOLD = 30.0

CONFIG_PATH = "neat_runtime_config.txt"

CHECKPOINT_PREFIX = "neat-checkpoint-"

BEST_GENOME_FILE = "best_genome.pkl"


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


if not os.path.exists(CONFIG_PATH):

    with open(CONFIG_PATH, "w") as f:
        f.write(NEAT_CONFIG)


# =========================================================
# CHECKPOINT
# =========================================================

def find_latest_checkpoint():

    checkpoints = []

    for file in os.listdir("."):

        if file.startswith(CHECKPOINT_PREFIX):

            try:
                generation = int(file.split("-")[-1])
                checkpoints.append((generation, file))
            except:
                pass

    if not checkpoints:
        return None

    checkpoints.sort()

    return checkpoints[-1][1]


# =========================================================
# RUNTIME
# =========================================================

class RuntimeEvaluator:

    def __init__(self):

        self.ws = None

    # =====================================================
    # CONNECT
    # =====================================================

    async def connect(self):

        self.ws = await websockets.connect(SERVER)

        await self.ws.send(json.dumps({

            "type": "hello",
            "name": BOT_NAME,
            "protocol": PROTOCOL_VERSION

        }))

        print("🟢 Connected")

    # =====================================================
    # STATE
    # =====================================================

    def build_state(self, data, angle):

        self_data = data["self"]

        wall = data.get(
            "wall_sensors",
            [0.0] * 3
        )

        enemy = data.get(
            "enemy_sensors",
            [0.0] * 3
        )

        return np.array([

            *wall,
            *enemy,

            self_data["hp"],

            math.sin(angle),
            math.cos(angle)

        ], dtype=np.float32)

    # =====================================================
    # RESPAWN
    # =====================================================

    def detect_respawn(
        self,
        prev_hp,
        hp,
        prev_x,
        prev_y,
        x,
        y
    ):

        hp_respawn = (

            prev_hp < RESPAWN_HP_THRESHOLD_LOW
            and
            hp > RESPAWN_HP_THRESHOLD_HIGH
        )

        dist = math.hypot(
            x - prev_x,
            y - prev_y
        )

        # teleport_respawn = (
        #     dist > RESPAWN_DISTANCE_THRESHOLD
        # )

        return hp_respawn #or teleport_respawn

    # =====================================================
    # REWARD
    # =====================================================

    def compute_reward(
        self,
        data,
        action,
        died,
        prev_score,
        idle_steps
    ):
        lifetime_penalty_step = 0.01 # Штраф за время жизни, чтобы поощрять более быстрые победы
        reward = 0.0

        self_data = data["self"]

        score = self_data["score"]

        wall = data["wall_sensors"]

        enemy = data["enemy_sensors"]

        delta_score = score - prev_score

        if delta_score > 0:

            reward += KILL_REWARD
            idle_steps = 0

        else:

            idle_steps += 1

        # reward += SURVIVAL_REWARD
        reward -= lifetime_penalty_step

        # reward += (
        #     max(enemy)
        #     * ENEMY_VISIBLE_REWARD
        # )

        # reward += (
        #     enemy[1]
        #     * AIM_REWARD
        # )

        # reward += (
        #     abs(action["move_front_back"])
        #     * MOVE_REWARD
        # )

        # reward += (
        #     abs(action["move_left_right"])
        #     * MOVE_REWARD
        # )

        # if wall[1] > 0.95:
        #     reward += WALL_PENALTY

        # reward += (
        #     abs(action["look_delta"])
        #     * SPIN_PENALTY
        # )

        if died:
            reward += DEATH_PENALTY

        reward = max(
            REWARD_CLAMP_MIN,
            min(reward, REWARD_CLAMP_MAX)
        )

        return reward, idle_steps

    # =====================================================
    # EVALUATE ONE GENOME
    # =====================================================

    async def evaluate_genome(
        self,
        genome,
        config,
        genome_id
    ):

        print(f"\n🎮 Genome {genome_id} started")

        net = neat.nn.RecurrentNetwork.create(
            genome,
            config
        )

        genome.fitness = 0.0

        angle = random.uniform(
            -math.pi,
            math.pi
        )

        prev_hp = 1.0

        prev_x = 0.0
        prev_y = 0.0

        prev_score = 0.0

        idle_steps = 0

        steps = 0

        while True:

            raw = await self.ws.recv()

            data = json.loads(raw)

            if data.get("type") != "bot_observation_v1":
                continue

            self_data = data["self"]

            hp = self_data["hp"]

            x = self_data["x"]
            y = self_data["y"]

            state = self.build_state(
                data,
                angle
            )

            outputs = net.activate(state)

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

            angle += (
                action["look_delta"]
                * ROTATION_SPEED
            )

            angle = math.atan2(
                math.sin(angle),
                math.cos(angle)
            )

            await self.ws.send(json.dumps({

                "move_front_back":
                    action["move_front_back"],

                "move_left_right":
                    action["move_left_right"],

                "angle":
                    angle,

                "shoot":
                    action["shoot"]
            }))

            died = self.detect_respawn(

                prev_hp,
                hp,

                prev_x,
                prev_y,

                x,
                y
            )

            reward, idle_steps = (
                self.compute_reward(

                    data,
                    action,
                    died,
                    prev_score,
                    idle_steps
                )
            )

            genome.fitness += reward

            steps += 1

            prev_hp = hp

            prev_x = x
            prev_y = y

            prev_score = self_data["score"]

            idle_timeout = (
                idle_steps >= MAX_IDLE_STEPS
            )

            if idle_timeout:

                genome.fitness += IDLE_PENALTY

                print("💤 Idle timeout")

            if (

                died
                or
                steps >= MAX_EPISODE_STEPS
                or
                idle_timeout
            ):

                print(
                    f"☠️ Genome {genome_id} finished | "
                    f"Fitness: {genome.fitness:.1f}"
                )

                return genome.fitness


# =========================================================
# MAIN ASYNC
# =========================================================

async def async_main():

    config = neat.Config(

        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,

        CONFIG_PATH
    )

    checkpoint = find_latest_checkpoint()

    if checkpoint:

        print(f"📦 Restoring {checkpoint}")

        population = neat.Checkpointer.restore_checkpoint(
            checkpoint
        )

    else:

        print("🌱 Fresh start")

        population = neat.Population(config)

    population.add_reporter(
        neat.StdOutReporter(True)
    )

    stats = neat.StatisticsReporter()

    population.add_reporter(stats)

    population.add_reporter(
        neat.Checkpointer(
            generation_interval=1,
            filename_prefix=CHECKPOINT_PREFIX
        )
    )

    runtime = RuntimeEvaluator()

    await runtime.connect()

    # =====================================================
    # STANDARD NEAT LOOP
    # =====================================================

    generation = 0

    while True:

        print(
            f"\n ****** Running generation {generation} ****** \n"
        )

        genomes = list(
            population.population.items()
        )

        for genome_id, genome in genomes:

            await runtime.evaluate_genome(
                genome,
                config,
                genome_id
            )

        best_genome = max(

            population.population.values(),

            key=lambda g:
                g.fitness
                if g.fitness is not None
                else -999999
        )

        print(
            f"\nBest fitness: "
            f"{best_genome.fitness:.1f}"
        )

        with open(
            BEST_GENOME_FILE,
            "wb"
        ) as f:

            pickle.dump(best_genome, f)

        # =================================================
        # REPRODUCE
        # =================================================

        population.reporters.start_generation(
            population.generation
        )

        population.reporters.post_evaluate(

            config,

            population.population,

            population.species,

            best_genome
        )

        population.population = (
            population.reproduction.reproduce(

                config,

                population.species,

                config.pop_size,

                population.generation
            )
        )

        if not population.population:

            population.reporters.complete_extinction()

            population.population = (
                population.reproduction.create_new(

                    config.genome_type,

                    config.genome_config,

                    config.pop_size
                )
            )

        population.species.speciate(

            config,

            population.population,

            population.generation
        )

        population.reporters.end_generation(

            config,

            population.population,

            population.species
        )

        population.generation += 1

        generation += 1


# =========================================================
# MAIN
# =========================================================

def main():

    asyncio.run(async_main())


if __name__ == "__main__":
    main()