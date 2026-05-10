import asyncio
import json
import math
import random
from dataclasses import dataclass, field

import websockets


"""
AI BOT для game
======================

Особенности:
- Подключается как обычный игрок.
- Анализирует состояние мира.
- Учится прямо во время игры.
- Хранит "веса" поведения.
- После каждого события изменяет стратегию.
- Может постепенно становиться агрессивнее или осторожнее.

Алгоритм:
-----------
Это не нейросеть, а online reinforcement learning.

Бот:
1. Оценивает состояние.
2. Выбирает действие.
3. Получает reward.
4. Меняет веса поведения.

Плюсы:
- Очень быстро.
- Не требует GPU.
- Можно запускать десятки ботов.
- Отлично подходит для серверной игры.

Запуск:
-------
python bot_ai_learning.py
"""

SERVER = "ws://127.0.0.1:8000/ws"


@dataclass
class Brain:
    """
    Простая обучаемая модель.

    Каждый параметр влияет на поведение.
    Во время игры веса меняются.
    """

    aggression: float = 1.0
    fear: float = 1.0
    accuracy: float = 1.0
    dodge: float = 1.0
    chase: float = 1.0

    learning_rate: float = 0.03

    def mutate_positive(self):
        """Усиливаем текущую стратегию."""

        self.aggression += random.uniform(0, self.learning_rate)
        self.accuracy += random.uniform(0, self.learning_rate)
        self.chase += random.uniform(0, self.learning_rate)

    def mutate_negative(self):
        """Делаем бота осторожнее."""

        self.fear += random.uniform(0, self.learning_rate)
        self.dodge += random.uniform(0, self.learning_rate)

        self.aggression *= 0.99

    def clamp(self):
        """Ограничиваем значения."""

        self.aggression = max(0.1, min(5.0, self.aggression))
        self.fear = max(0.1, min(5.0, self.fear))
        self.accuracy = max(0.1, min(5.0, self.accuracy))
        self.dodge = max(0.1, min(5.0, self.dodge))
        self.chase = max(0.1, min(5.0, self.chase))


class AIBot:

    def __init__(self):

        self.my_id = None
        self.brain = Brain()

        self.prev_hp = 100
        self.prev_score = 0

        self.ticks_alive = 0

        self.last_move_change = 0
        self.random_strafe_dir = 1

    async def run(self):

        async with websockets.connect(SERVER) as ws:

            print("AI BOT CONNECTED")

            while True:

                try:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if data.get("type") == "welcome":
                        self.my_id = data["player_id"]
                        print("MY ID:", self.my_id)

                        await ws.send(json.dumps({
                            "type": "hello",
                            "name": "AdaptiveAI"
                        }))
                        continue

                    await self.process_state(ws, data)

                except Exception as e:
                    print("AI BOT ERROR:", e)
                    break

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
            print("BOT DEAD")
            return

        self.learn(me)

        nearest_enemy = self.find_nearest(me, enemies)
        nearest_player = self.find_nearest(
            me,
            [p for p in players if p["id"] != self.my_id]
        )

        target = self.choose_target(me, nearest_enemy, nearest_player)

        dx = 0
        dy = 0
        angle = me.get("angle", 0)
        shoot = False

        if target:

            tx = target["x"] - me["x"]
            ty = target["y"] - me["y"]

            dist = math.hypot(tx, ty)

            if dist > 0:

                angle = math.atan2(ty, tx)

                # === ОБУЧАЕМОЕ ДВИЖЕНИЕ ===

                preferred_distance = 120 * self.brain.fear

                # Агрессивный пуш
                if dist > preferred_distance:
                    dx += (tx / dist) * self.brain.chase
                    dy += (ty / dist) * self.brain.chase

                # Отступление
                if dist < 70 * self.brain.fear:
                    dx -= (tx / dist) * self.brain.fear
                    dy -= (ty / dist) * self.brain.fear

                # Стрейф
                self.last_move_change += 1

                if self.last_move_change > 20:
                    self.random_strafe_dir *= -1
                    self.last_move_change = 0

                strafe_x = -ty / dist
                strafe_y = tx / dist

                dx += strafe_x * 0.8 * self.brain.dodge * self.random_strafe_dir
                dy += strafe_y * 0.8 * self.brain.dodge * self.random_strafe_dir

                # Стрельба
                hit_chance = self.estimate_hit_probability(dist)

                if hit_chance > 0.3:
                    shoot = True

        # === УКЛОНЕНИЕ ОТ ПУЛЬ ===

        dodge_x = 0
        dodge_y = 0

        for bullet in bullets:

            future_x = bullet["x"] + bullet["vx"] * 5
            future_y = bullet["y"] + bullet["vy"] * 5

            bdx = future_x - me["x"]
            bdy = future_y - me["y"]

            danger = math.hypot(bdx, bdy)

            if danger < 80:

                perp_x = -bdy
                perp_y = bdx

                length = math.hypot(perp_x, perp_y)

                if length > 0:
                    dodge_x += (perp_x / length) * self.brain.dodge
                    dodge_y += (perp_y / length) * self.brain.dodge

        dx += dodge_x
        dy += dodge_y

        # Нормализация
        length = math.hypot(dx, dy)

        if length > 0:
            dx /= length
            dy /= length

        action = {
            "dx": dx,
            "dy": dy,
            "shoot": shoot,
            "angle": angle
        }

        await ws.send(json.dumps(action))

        self.ticks_alive += 1

    def find_nearest(self, me, targets):

        nearest = None
        nearest_dist = 999999

        for t in targets:

            dx = t["x"] - me["x"]
            dy = t["y"] - me["y"]

            dist = math.hypot(dx, dy)

            if dist < nearest_dist:
                nearest = t
                nearest_dist = dist

        return nearest

    def choose_target(self, me, enemy, player):
        """
        Выбираем цель.

        Со временем бот сам начинает предпочитать
        либо игроков, либо монстров.
        """

        if enemy and player:

            if self.brain.aggression > 1.4:
                return player

            return enemy

        return player or enemy

    def estimate_hit_probability(self, dist):
        """
        Оценка вероятности попадания.
        """

        value = self.brain.accuracy * (1.0 / max(1.0, dist / 100))

        return max(0.0, min(1.0, value))

    def learn(self, me):
        """
        Онлайн-обучение.

        Reward система:
        + очки = хорошо
        - потеря HP = плохо
        """

        hp_delta = me["hp"] - self.prev_hp
        score_delta = me["score"] - self.prev_score

        reward = 0

        # Нанесли урон / убили
        if score_delta > 0:
            reward += score_delta * 0.5

        # Получили урон
        if hp_delta < 0:
            reward += hp_delta * 1.2

        # Долго живем
        reward += 0.02

        # === ОБУЧЕНИЕ ===

        if reward > 0:
            self.brain.mutate_positive()
        else:
            self.brain.mutate_negative()

        self.brain.clamp()

        self.prev_hp = me["hp"]
        self.prev_score = me["score"]

        if self.ticks_alive % 300 == 0:
            self.print_stats(reward)

    def print_stats(self, reward):

        print("=" * 60)
        print("AI STATS")
        print(f"reward:      {reward:.2f}")
        print(f"aggression:  {self.brain.aggression:.2f}")
        print(f"fear:        {self.brain.fear:.2f}")
        print(f"accuracy:    {self.brain.accuracy:.2f}")
        print(f"dodge:       {self.brain.dodge:.2f}")
        print(f"chase:       {self.brain.chase:.2f}")

        print(f"prev_score:       {self.prev_score:.2f}")
        print("=" * 60)


async def main():

    bot = AIBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
