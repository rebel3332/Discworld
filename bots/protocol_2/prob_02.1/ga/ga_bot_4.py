# ga_bot_4.py
# =========================================================
# Genetic MLP Bot
# =========================================================
# Features:
# - Persistent champion loading/saving
# - Proper evolution continuation after restart
# - Configurable constants
# - Relative movement controls
# - Hidden layer MLP
# - Champion elitism
# - Stable mutation pipeline
# =========================================================

import asyncio
import json
import math
import random
import time
import os
import pickle

import numpy as np
import websockets


# =========================================================
# CONFIG
# =========================================================

SERVER = "ws://localhost:8000/ws"

PROTOCOL_VERSION = 2.1

BOT_NAME = "2.1_GA_MLP_4"

# ---------------------------------------------------------
# NETWORK
# ---------------------------------------------------------

INPUT_SIZE = 11
HIDDEN_SIZE = 100
OUTPUT_SIZE = 4

# ---------------------------------------------------------
# EVOLUTION
# ---------------------------------------------------------

POPULATION_SIZE = 1000

MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.15

# ---------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------

ROTATION_SPEED = 0.15

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

BEST_GENOME_FILE = "ga_bot_4_best_genome.npy"
CHECKPOINT_FILE = "ga_bot_4_population.pkl"

# ---------------------------------------------------------
# FITNESS
# ---------------------------------------------------------

USE_SURVIVAL_TIME = False
SURVIVAL_TIME_MULTIPLIER = 2.0

# ---------------------------------------------------------
# RESPAWN DETECTION
# ---------------------------------------------------------

RESPAWN_HP_THRESHOLD_LOW = 0.3
RESPAWN_HP_THRESHOLD_HIGH = 0.8

RESPAWN_DISTANCE_THRESHOLD = 30.0

# ---------------------------------------------------------
# RANDOM INIT
# ---------------------------------------------------------

USE_RANDOM_START_ANGLE = True


# =========================================================
# GENOME
# =========================================================

class Genome:

    def __init__(
        self,
        in_dim=INPUT_SIZE,
        hidden_dim=HIDDEN_SIZE,
        out_dim=OUTPUT_SIZE,
        weights=None
    ):

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fitness = 0.0

        if weights is not None:

            self.w1, self.w2 = weights

        else:

            # Xavier init
            limit1 = np.sqrt(
                6.0 / (in_dim + hidden_dim)
            )

            limit2 = np.sqrt(
                6.0 / (hidden_dim + 1 + out_dim)
            )

            self.w1 = np.random.uniform(
                -limit1,
                limit1,
                (in_dim, hidden_dim)
            )

            self.w2 = np.random.uniform(
                -limit2,
                limit2,
                (hidden_dim + 1, out_dim)
            )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, observation):

        wall = observation.get(
            "wall_sensors",
            [0.0] * 3
        )

        enemy = observation.get(
            "enemy_sensors",
            [0.0] * 3
        )
        additional = observation.get(
            "additional_info",
            {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0}
        )

        inputs = np.array(
            [min(max(v, 0.0), 1.0) for v in wall]
            +
            [min(max(v, 0.0), 1.0) for v in enemy]
            +
            [min(max(v, 0.0), 1.0) for v in additional.values()]
            +
            [1.0]  # bias
        )

        # -------------------------------------------------
        # Hidden
        # -------------------------------------------------

        hidden = np.dot(
            inputs,
            self.w1
        )

        hidden = np.tanh(hidden)

        hidden = np.concatenate(
            [hidden, [1.0]]
        )

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        outputs = np.dot(
            hidden,
            self.w2
        )

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

        return {

            "move_front_back":
                move_front_back,

            "move_left_right":
                move_left_right,

            "look_delta":
                look_delta,

            "shoot":
                shoot
        }

    # =====================================================
    # MUTATION
    # =====================================================

    def mutate(
        self,
        rate=MUTATION_RATE,
        strength=MUTATION_STRENGTH
    ):

        mask1 = (
            np.random.rand(*self.w1.shape)
            < rate
        )

        self.w1 += (
            mask1
            *
            np.random.randn(*self.w1.shape)
            *
            strength
        )

        mask2 = (
            np.random.rand(*self.w2.shape)
            < rate
        )

        self.w2 += (
            mask2
            *
            np.random.randn(*self.w2.shape)
            *
            strength
        )

        return self

    # =====================================================
    # SAVE / LOAD
    # =====================================================

    def save(self, path):

        np.save(
            path,
            np.array(
                [self.w1, self.w2],
                dtype=object
            )
        )

        print(f"💾 Saved genome: {path}")


    @staticmethod
    def load(path):

        w1, w2 = np.load(
            path,
            allow_pickle=True
        )

        genome = Genome(
            weights=(w1, w2)
        )

        return genome
    



# =========================================================
# POPULATION
# =========================================================

class Population:

    def __init__(self, size=POPULATION_SIZE):

        self.size = size

        self.individuals = [
            Genome()
            for _ in range(size)
        ]

        self.generation = 1

        self.best_ever = -999999

        # ---------------------------------------------
        # LOAD CHAMPION
        # ---------------------------------------------

        if os.path.exists(BEST_GENOME_FILE):

            try:

                # champion = Genome.load(
                #     BEST_GENOME_FILE
                # )

                # self.individuals[0] = champion

                # print(
                #     f"📦 Loaded champion from "
                #     f"{BEST_GENOME_FILE}"
                # )


                if os.path.exists(CHECKPOINT_FILE):

                    self.load_checkpoint()

                else:

                    print("🌱 Fresh evolution start")

            except Exception as e:

                print(
                    f"❌ Failed to load champion: {e}"
                )

        else:

            print("🌱 Fresh evolution start")

        # ---------------------------------------------
        # SORT
        # ---------------------------------------------

        self.individuals.sort(
            key=lambda g: g.fitness,
            reverse=True
        )

        # ---------------------------------------------
        # START WITH CHAMPION
        # ---------------------------------------------

        self.current_genome = self.individuals[0]

    # =====================================================
    # ACTIVE GENOME
    # =====================================================

    def get_next_genome(self):

        return self.current_genome

    # =====================================================
    # REPORT
    # =====================================================

    def report_life_end(
        self,
        genome,
        fitness,
        duration
    ):

        genome.fitness = fitness

        # # Добавляем старение всех геномов, чтобы избежать застоя в локальном оптимуме из-за элитизма
        # for i, g in enumerate(self.individuals): 
        #     if self.individuals[i].fitness > 0:
        #         self.individuals[i].fitness *= 0.99

        scores = [
            g.fitness
            for g in self.individuals
        ]

        best = max(scores)
        avg = sum(scores) / len(scores)
        worst = min(scores)

        print(
            f"📊 "
            f"Gen {self.generation:4d} | "
            f"Best: {best:7.1f} | "
            f"Avg: {avg:7.1f} | "
            f"Worst: {worst:7.1f} | "
            f"Time: {duration:.1f}s"
        )

        # -------------------------------------------------
        # NEW RECORD
        # -------------------------------------------------

        if fitness > self.best_ever:

            self.best_ever = fitness

            print(
                f"🏆 NEW RECORD: {fitness:.1f}"
            )

            genome.save(BEST_GENOME_FILE)

        # -------------------------------------------------
        # SORT
        # -------------------------------------------------

        self.individuals.sort(
            key=lambda g: g.fitness,
            reverse=True
        )

        self.individuals[0].fitness *= 0.9999 # Лёгкий спад для чемпиона, чтобы поощрять эволюцию

        champion = self.individuals[0]

        # -------------------------------------------------
        # CHILD
        # -------------------------------------------------

        child = Genome(
            weights=(
                champion.w1.copy(),
                champion.w2.copy()
            )
        ).mutate()

        # Replace worst
        self.individuals[-1] = child

        # Active genome = child
        self.current_genome = child

        self.generation += 1
        self.save_checkpoint()


    def save_checkpoint(self):

        data = {

            "generation":
                self.generation,

            "best_ever":
                self.best_ever,

            "individuals": [

                {
                    "w1": g.w1,
                    "w2": g.w2,
                    "fitness": g.fitness
                }

                for g in self.individuals
            ]
        }

        try:
            with open(CHECKPOINT_FILE, "wb") as f:
                pickle.dump(data, f)
            print(
                f"💾 Saved checkpoint "
                f"(gen {self.generation})"
            )       
        except Exception as e:
            print(f"Error saving checkpoint: {e}")



    def load_checkpoint(self):

        with open(CHECKPOINT_FILE, "rb") as f:

            data = pickle.load(f)

        self.generation = data["generation"]

        self.best_ever = data["best_ever"]

        self.individuals = []

        for g in data["individuals"]:

            genome = Genome(
                weights=(
                    g["w1"],
                    g["w2"]
                )
            )

            genome.fitness = g["fitness"]

            self.individuals.append(genome)

        print(
            f"📦 Loaded checkpoint "
            f"(gen {self.generation})"
        )


# =========================================================
# BOT
# =========================================================

class GeneticBot:

    def __init__(self, population):

        self.pop = population

        self.genome = population.get_next_genome()

        self.angle = (
            random.uniform(
                -math.pi,
                math.pi
            )
            if USE_RANDOM_START_ANGLE
            else 0.0
        )

        self.life_start_time = 0.0

        self.life_active = False

        self.prev_hp = 1.0
        self.prev_x = 0.0
        self.prev_y = 0.0

        self.last_score = 0.0
        self.lifetime_penalty_step = 0.01 # Штраф за время жизни, чтобы поощрять более быстрые победы
        self.lifetime_penalty_score = 0.0 # Накопленный штраф за время жизни особи

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
        ) and self.life_active

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
                # RESPAWN
                # -----------------------------------------

                # print(f"Прогноз награды: {self.last_score - self.lifetime_penalty_score:.3f}")
                if self.detect_respawn(hp, x, y):

                    duration = (
                        time.time()
                        -
                        self.life_start_time
                    )

                    # fitness = score
                    fitness = self.last_score - self.lifetime_penalty_score
                    self.lifetime_penalty_score = 0.0

                    if USE_SURVIVAL_TIME:
                        fitness += (
                            duration * SURVIVAL_TIME_MULTIPLIER
                        )

                    self.pop.report_life_end(
                        self.genome,
                        fitness,
                        duration
                    )

                    self.genome = (
                        self.pop.get_next_genome()
                    )

                    self.life_start_time = (
                        time.time()
                    )

                    if USE_RANDOM_START_ANGLE:
                        self.angle = random.uniform(
                            -math.pi,
                            math.pi
                        )

                elif not self.life_active:
                    self.life_active = True
                    self.life_start_time = (
                        time.time()
                    )
                else:
                    # Начисляем штраф за время жизни
                    self.lifetime_penalty_score += self.lifetime_penalty_step

                # -----------------------------------------
                # SAVE PREVIOUS
                # -----------------------------------------

                self.prev_hp = hp
                self.prev_x = x
                self.prev_y = y

                self.last_score = score

                # -----------------------------------------
                # THINK
                # -----------------------------------------

                action = self.genome.forward(data)

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

                self.last_score = score


# =========================================================
# MAIN
# =========================================================

async def main():

    population = Population()

    bot = GeneticBot(population)

    await bot.run_forever()


if __name__ == "__main__":

    asyncio.run(main())