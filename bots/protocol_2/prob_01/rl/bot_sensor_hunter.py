# bot_sensor_hunter.py
# Простой бот, который использует сенсоры для охоты на врагов.
import asyncio
import json
import math
import random

import websockets


SERVER = "ws://localhost:8000/ws"

PROTOCOL_VERSION = 2


class SensorHunterBot:

    def __init__(self):

        self.angle = random.uniform(
            -math.pi,
            math.pi
        )

        self.turn_speed = 0.15

        self.move_speed = 1.0

        self.last_enemy_seen = 0

    # =====================================================
    # HELPERS
    # =====================================================

    def sensor_to_vector(self, sensors):

        """
        Преобразуем лучи в steering vector.

        Чем ближе препятствие —
        тем сильнее отталкивание.
        """

        steer_x = 0
        steer_y = 0

        sensor_count = len(sensors)

        for i, dist in enumerate(sensors):

            angle = (
                i / sensor_count
            ) * math.pi * 2

            # obstacle force
            strength = 1.0 - dist

            steer_x -= math.cos(angle) * strength
            steer_y -= math.sin(angle) * strength

        return steer_x, steer_y

    def find_enemy_direction(self, enemy_sensors):

        """
        Ищем направление,
        где враг ближе всего.
        """

        best_index = None
        best_dist = 999

        for i, dist in enumerate(enemy_sensors):

            if dist < best_dist:

                best_dist = dist
                best_index = i

        if best_index is None:
            return None, None

        sensor_count = len(enemy_sensors)

        angle = (
            best_index / sensor_count
        ) * math.pi * 2

        return angle, best_dist

    # =====================================================
    # AI
    # =====================================================

    def think(
        self,
        wall_sensors,
        enemy_sensors
    ):

        # =========================================
        # WALL AVOIDANCE
        # =========================================

        avoid_x, avoid_y = self.sensor_to_vector(
            wall_sensors
        )

        # =========================================
        # ENEMY SEEKING
        # =========================================

        enemy_angle, enemy_dist = (
            self.find_enemy_direction(
                enemy_sensors
            )
        )

        shoot = False

        # =========================================
        # TARGET ENEMY
        # =========================================

        if (
            enemy_angle is not None
            and enemy_dist < 1.0
        ):

            self.last_enemy_seen = 30

            # steering toward enemy
            target_angle = enemy_angle

            angle_diff = (
                target_angle - self.angle
            )

            # normalize
            while angle_diff > math.pi:
                angle_diff -= math.pi * 2

            while angle_diff < -math.pi:
                angle_diff += math.pi * 2

            self.angle += (
                angle_diff * self.turn_speed
            )

            # shoot if centered
            if abs(angle_diff) < 0.15:
                shoot = True

        else:

            # =====================================
            # RANDOM WANDER
            # =====================================

            self.last_enemy_seen -= 1

            self.angle += random.uniform(
                -0.05,
                0.05
            )

        # =========================================
        # COMBINE MOVEMENT
        # =========================================

        move_x = math.cos(self.angle)
        move_y = math.sin(self.angle)

        # obstacle avoidance
        move_x += avoid_x * 1.8
        move_y += avoid_y * 1.8

        # normalize
        length = math.hypot(
            move_x,
            move_y
        )

        if length > 0:

            move_x /= length
            move_y /= length

        return {
            "dx": move_x,
            "dy": move_y,
            "angle": self.angle,
            "shoot": shoot
        }


# =========================================================
# NETWORK
# =========================================================

async def run_bot():

    bot = SensorHunterBot()

    async with websockets.connect(SERVER) as ws:

        # hello packet
        await ws.send(json.dumps({
            "type": "hello",
            "name": "SensorHunter",
            "protocol": PROTOCOL_VERSION
        }))

        while True:

            raw = await ws.recv()

            data = json.loads(raw)
            print(data)
            if data.get("type") != "bot_observation_v1":
                continue

            wall_sensors = data.get(
                "wall_sensors",
                []
            )

            enemy_sensors = data.get(
                "enemy_sensors",
                []
            )

            command = bot.think(
                wall_sensors,
                enemy_sensors
            )

            await ws.send(
                json.dumps(command)
            )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    asyncio.run(run_bot())

