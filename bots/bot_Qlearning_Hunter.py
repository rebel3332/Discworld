import asyncio
import json
import math
import random
from collections import defaultdict

import websockets


"""
Q-Learning Bot
===============

Новый тип AI-бота.
Классический Q-learning с таблицей состояний.

В отличие от предыдущего adaptive-бота,
этот использует:

- состояния (state)
- действия (actions)
- reward
- Q-table

То есть это уже настоящий Reinforcement Learning.

Бот постепенно учится:
- когда атаковать
- когда убегать
- когда стрейфиться
- когда пушить игроков

Особенность:
-------------
Никаких нейросетей.
Все работает на обычной Q-table.

Это:
- быстро
- стабильно
- идеально для realtime игры
- можно запускать десятки ботов
"""

SERVER = "ws://127.0.0.1:8000/ws"
# SERVER = "ws://216.24.57.7:80/ws" #discworld.onrender.com



class QLearningBot:

    ACTIONS = [
        "attack",
        "retreat",
        "strafe_left",
        "strafe_right",
        "idle"
    ]

    def __init__(self):

        self.my_id = None

        # Q[state][action]
        self.q_table = defaultdict(lambda: defaultdict(float))

        self.learning_rate = 0.15
        self.discount = 0.92
        self.epsilon = 0.20

        self.prev_state = None
        self.prev_action = None

        self.prev_hp = 100
        self.prev_score = 0

        self.ticks = 0

    async def run(self):

        async with websockets.connect(SERVER) as ws:

            print("Q-LEARNING BOT CONNECTED")

            while True:

                try:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if data.get("type") == "welcome":
                        self.my_id = data["player_id"]
                        print("BOT ID:", self.my_id)

                        await ws.send(json.dumps({
                            "type": "hello",
                            "name": "QHunter"
                        }))
                        continue

                    await self.process_state(ws, data)

                except Exception as e:
                    print("BOT ERROR:", e)
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
            return

        target = self.find_best_target(me, enemies, players)

        game_state = self.build_state(me, target, bullets)

        action = self.choose_action(game_state)

        reward = self.calculate_reward(me)

        self.learn(
            self.prev_state,
            self.prev_action,
            reward,
            game_state
        )

        dx, dy, shoot, angle = self.execute_action(
            action,
            me,
            target,
            bullets
        )

        payload = {
            "dx": dx,
            "dy": dy,
            "shoot": shoot,
            "angle": angle
        }

        await ws.send(json.dumps(payload))

        self.prev_state = game_state
        self.prev_action = action

        self.prev_hp = me["hp"]
        self.prev_score = me["score"]

        self.ticks += 1

        if self.ticks % 300 == 0:
            self.debug_stats(game_state)

    def build_state(self, me, target, bullets):
        """
        Создаем дискретное состояние.

        RL любит компактные state.
        """

        hp_bucket = self.bucket_hp(me["hp"])

        enemy_distance = "none"

        if target:
            dist = math.hypot(
                target["x"] - me["x"],
                target["y"] - me["y"]
            )

            if dist < 80:
                enemy_distance = "close"
            elif dist < 180:
                enemy_distance = "mid"
            else:
                enemy_distance = "far"

        bullet_danger = "safe"

        for bullet in bullets:

            dist = math.hypot(
                bullet["x"] - me["x"],
                bullet["y"] - me["y"]
            )

            if dist < 70:
                bullet_danger = "danger"
                break

        return (
            hp_bucket,
            enemy_distance,
            bullet_danger
        )

    def bucket_hp(self, hp):

        if hp < 30:
            return "low"

        if hp < 70:
            return "mid"

        return "high"

    def choose_action(self, state):
        """
        Epsilon-greedy policy.
        """

        # exploration
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)

        # exploitation
        q_values = self.q_table[state]

        best_action = None
        best_value = -999999

        for action in self.ACTIONS:

            value = q_values[action]

            if value > best_value:
                best_value = value
                best_action = action

        return best_action or random.choice(self.ACTIONS)

    def learn(self, prev_state, prev_action, reward, next_state):
        """
        Классический Q-learning.
        """

        if prev_state is None or prev_action is None:
            return

        old_q = self.q_table[prev_state][prev_action]

        next_max_q = max(
            self.q_table[next_state][a]
            for a in self.ACTIONS
        )

        new_q = old_q + self.learning_rate * (
            reward + self.discount * next_max_q - old_q
        )

        self.q_table[prev_state][prev_action] = new_q

    def calculate_reward(self, me):

        reward = 0

        score_gain = me["score"] - self.prev_score
        hp_loss = self.prev_hp - me["hp"]

        # убийства / попадания
        reward += score_gain * 1.5

        # урон по нам
        reward -= hp_loss * 2.0

        # бонус за выживание
        reward += 0.05

        return reward

    def execute_action(self, action, me, target, bullets):

        dx = 0
        dy = 0
        shoot = False
        angle = me.get("angle", 0)

        if not target:
            return dx, dy, shoot, angle

        tx = target["x"] - me["x"]
        ty = target["y"] - me["y"]

        dist = math.hypot(tx, ty)

        if dist <= 0:
            return dx, dy, shoot, angle

        angle = math.atan2(ty, tx)

        nx = tx / dist
        ny = ty / dist

        if action == "attack":

            dx = nx
            dy = ny
            shoot = True

        elif action == "retreat":

            dx = -nx
            dy = -ny
            shoot = True

        elif action == "strafe_left":

            dx = -ny
            dy = nx
            shoot = True

        elif action == "strafe_right":

            dx = ny
            dy = -nx
            shoot = True

        elif action == "idle":

            shoot = dist < 140

        # уклонение от пуль
        avoid_x, avoid_y = self.avoid_bullets(me, bullets)

        dx += avoid_x
        dy += avoid_y

        # нормализация
        length = math.hypot(dx, dy)

        if length > 0:
            dx /= length
            dy /= length

        return dx, dy, shoot, angle

    def avoid_bullets(self, me, bullets):

        move_x = 0
        move_y = 0

        for bullet in bullets:

            dx = bullet["x"] - me["x"]
            dy = bullet["y"] - me["y"]

            dist = math.hypot(dx, dy)

            if dist < 60:

                perp_x = -dy
                perp_y = dx

                length = math.hypot(perp_x, perp_y)

                if length > 0:
                    move_x -= perp_x / length
                    move_y -= perp_y / length

        return move_x, move_y

    def find_best_target(self, me, enemies, players):

        targets = []

        targets.extend(enemies)

        for p in players:
            if p["id"] != self.my_id:
                targets.append(p)

        best = None
        best_score = -999999

        for t in targets:

            dist = math.hypot(
                t["x"] - me["x"],
                t["y"] - me["y"]
            )

            score = 1.0 / max(1, dist)

            # игроков любит сильнее
            if "hp" in t:
                score *= 1.5

            if score > best_score:
                best_score = score
                best = t

        return best

    def debug_stats(self, state):

        print("=" * 60)
        print("Q LEARNING STATS")
        print("state:", state)
        print("epsilon:", round(self.epsilon, 3))

        qvals = self.q_table[state]

        for action in self.ACTIONS:
            print(f"{action:15s}: {qvals[action]:.3f}")

        print(f"Score          : {self.prev_score}")
        print("=" * 60)


async def main():

    bot = QLearningBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())