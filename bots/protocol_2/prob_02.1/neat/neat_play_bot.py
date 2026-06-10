# neat_play_bot.py
# =========================================================
# NEAT PLAY BOT
# =========================================================
# Loads trained genome and only plays
#
# FEATURES
# ---------------------------------------------------------
# - Loads best genome
# - No evolution
# - Single persistent websocket
# - Recurrent NEAT support
# - Protocol v2
# - Relative movement
# - Stable realtime inference
# =========================================================

import asyncio
import json
import math
import pickle
import random

import neat
import numpy as np
import websockets


# =========================================================
# CONFIG
# =========================================================

SERVER = "ws://localhost:8000/ws"

BOT_NAME = "NEAT_PLAYER"

PROTOCOL_VERSION = 2

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

CONFIG_PATH = "neat_runtime_config.txt"

GENOME_FILE = "best_genome.pkl"

# ---------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------

ROTATION_SPEED = 0.15

USE_RANDOM_START_ANGLE = True


# =========================================================
# LOAD CONFIG
# =========================================================

config = neat.Config(

    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,

    CONFIG_PATH
)


# =========================================================
# LOAD GENOME
# =========================================================

print(f"📦 Loading genome: {GENOME_FILE}")

with open(GENOME_FILE, "rb") as f:

    genome = pickle.load(f)

print("✅ Genome loaded")


# =========================================================
# CREATE NETWORK
# =========================================================

net = neat.nn.RecurrentNetwork.create(
    genome,
    config
)

print("🧠 Network created")


# =========================================================
# BOT
# =========================================================

class NeatPlayBot:

    def __init__(self):

        self.angle = (

            random.uniform(
                -math.pi,
                math.pi
            )

            if USE_RANDOM_START_ANGLE

            else 0.0
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

                # -------------------------------------------------
                # BUILD STATE
                # -------------------------------------------------

                state = self.build_state(data)

                # -------------------------------------------------
                # NETWORK
                # -------------------------------------------------

                outputs = net.activate(state)

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


# =========================================================
# MAIN
# =========================================================

async def main():

    bot = NeatPlayBot()

    await bot.run()


if __name__ == "__main__":

    asyncio.run(main())