"""Этот бот использует:

* Genetic Algorithm
* population
* mutation
* elite selection
* fixed neural policy
"""



import asyncio
import copy
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import websockets
import os


SERVER = "ws://127.0.0.1:8000/ws"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_SIZE = 16
ACTION_SIZE = 6

POPULATION_SIZE = 100
ELITE_COUNT = 6
MUTATION_POWER = 0.02
EPISODE_TICKS = 3000
MAX_AGENT_LIFETIME = 300

SAVE_FILE = "bot_genetic_population.pt"

# =========================================================
# NETWORK
# =========================================================

class Brain(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, x):

        return self.net(x)


# =========================================================
# AGENT
# =========================================================

class Agent:

    def __init__(self):

        self.brain = Brain().to(DEVICE)

        self.fitness = 0.0

        self.reset_stats()

    def reset_stats(self):

        self.damage_done = 0
        self.damage_taken = 0

        self.enemy_hits = 0
        self.player_hits = 0

        self.survival_ticks = 0

        self.kills = 0

    def clone(self):

        clone = Agent()

        clone.brain.load_state_dict(
            copy.deepcopy(
                self.brain.state_dict()
            )
        )

        return clone


# =========================================================
# EVOLUTION
# =========================================================

class GeneticTrainer:

    def __init__(self):

        self.population = [
            Agent()
            for _ in range(POPULATION_SIZE)
        ]

        self.generation = 0

        self.load_population()

    def mutate(self, agent):

        with torch.no_grad():

            for param in agent.brain.parameters():

                noise = torch.randn_like(param)

                param += noise * MUTATION_POWER

    def crossover(self, parent_a, parent_b):

        child = Agent()

        with torch.no_grad():

            for child_param, a_param, b_param in zip(
                child.brain.parameters(),
                parent_a.brain.parameters(),
                parent_b.brain.parameters()
            ):

                mask = torch.rand_like(child_param) > 0.5

                child_param.copy_(
                    torch.where(mask, a_param, b_param)
                )

        return child

    def save_population(self):

        data = {
            "generation": self.generation,
            "population": [
                agent.brain.state_dict()
                for agent in self.population
            ]
        }

        torch.save(data, SAVE_FILE)

        print("POPULATION SAVED")

    def load_population(self):

        if not os.path.exists(SAVE_FILE):

            print("NEW POPULATION CREATED")

            return

        try:

            data = torch.load(
                SAVE_FILE,
                map_location=DEVICE
            )

            self.generation = data["generation"]

            saved_population = data["population"]

            for agent, weights in zip(
                self.population,
                saved_population
            ):

                agent.brain.load_state_dict(weights)

            print(
                "POPULATION LOADED | GENERATION:",
                self.generation
            )

        except Exception as e:

            print("LOAD ERROR:", e)

    def evolve(self):

        self.population.sort(
            key=lambda a: a.fitness,
            reverse=True
        )

        best = self.population[0]

        print(
            "GENERATION:",
            self.generation,
            "BEST FITNESS:",
            round(best.fitness, 2)
        )

        next_population = []

        elites = self.population[:ELITE_COUNT]

        for elite in elites:
            next_population.append(elite.clone())

        while len(next_population) < POPULATION_SIZE:

            parent_a = random.choice(elites)
            parent_b = random.choice(elites)

            child = self.crossover(
                parent_a,
                parent_b
            )

            self.mutate(child)

            next_population.append(child)

        self.population = next_population

        self.generation += 1

        self.save_population()


# =========================================================
# BOT
# =========================================================

class GeneticBot:

    def __init__(self, agent):

        self.agent = agent

        self.my_id = None

        self.prev_hp = 100
        self.prev_enemy_hits = 0
        self.prev_player_hits = 0

        self.ticks = 0

    async def run(self):

        async with websockets.connect(SERVER) as ws:

            await ws.send(json.dumps({
                "type": "hello",
                "name": "GeneticBot"
            }))

            while True:

                try:

                    msg = await ws.recv()

                    data = json.loads(msg)

                    if data.get("type") == "welcome":

                        self.my_id = data["player_id"]

                        continue

                    done = await self.process_state(ws, data)

                    if done:
                        break

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
            return False

        obs = self.build_state(
            me,
            players,
            enemies,
            bullets
        )

        action = self.select_action(obs)

        dx, dy, shoot, angle = self.decode_action(
            action,
            me,
            players,
            enemies
        )

        await ws.send(json.dumps({
            "dx": dx,
            "dy": dy,
            "shoot": shoot,
            "angle": angle
        }))

        self.calculate_fitness(me)

        self.ticks += 1

        # агент принудительно завершается
        # если живет слишком долго
        if self.ticks >= MAX_AGENT_LIFETIME:

            print(
                "AGENT TIME LIMIT REACHED:",
                self.agent.fitness
            )

            return True

        return self.ticks >= EPISODE_TICKS

    def build_state(self, me, players, enemies, bullets):

        nearest_enemy_dx = 0
        nearest_enemy_dy = 0
        nearest_enemy_dist = 0

        best_dist = 999999

        for e in enemies:

            dx = e["x"] - me["x"]
            dy = e["y"] - me["y"]

            dist = math.hypot(dx, dy)

            if dist < best_dist:

                best_dist = dist

                nearest_enemy_dx = dx / 800
                nearest_enemy_dy = dy / 600
                nearest_enemy_dist = dist / 1000

        nearest_player_dx = 0
        nearest_player_dy = 0

        best_dist = 999999

        for p in players:

            if p["id"] == self.my_id:
                continue

            dx = p["x"] - me["x"]
            dy = p["y"] - me["y"]

            dist = math.hypot(dx, dy)

            if dist < best_dist:

                best_dist = dist

                nearest_player_dx = dx / 800
                nearest_player_dy = dy / 600

        bullet_dx = 0
        bullet_dy = 0
        bullet_dist = 0

        best_dist = 999999

        for b in bullets:

            dx = b["x"] - me["x"]
            dy = b["y"] - me["y"]

            dist = math.hypot(dx, dy)

            if dist < best_dist:

                best_dist = dist

                bullet_dx = dx / 800
                bullet_dy = dy / 600
                bullet_dist = dist / 1000

        return np.array([

            me["hp"] / 100,

            me["x"] / 800,
            me["y"] / 600,

            nearest_enemy_dx,
            nearest_enemy_dy,
            nearest_enemy_dist,

            nearest_player_dx,
            nearest_player_dy,

            bullet_dx,
            bullet_dy,
            bullet_dist,

            len(enemies) / 20,
            len(players) / 10,
            len(bullets) / 30,

            me["enemy_hits"] / 20,
            me["player_hits"] / 20,

        ], dtype=np.float32)

    def select_action(self, obs):

        state = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            logits = self.agent.brain(state)

        probs = torch.softmax(logits, dim=-1)

        action = torch.argmax(probs).item()

        return action

    def decode_action(self, action, me, players, enemies):

        targets = []

        targets.extend(enemies)

        for p in players:
            if p["id"] != self.my_id:
                targets.append(p)

        if not targets:
            return 0, 0, False, 0

        target = min(
            targets,
            key=lambda t: math.hypot(
                t["x"] - me["x"],
                t["y"] - me["y"]
            )
        )

        dx = target["x"] - me["x"]
        dy = target["y"] - me["y"]

        dist = math.hypot(dx, dy)

        if dist <= 0:
            return 0, 0, False, 0

        nx = dx / dist
        ny = dy / dist

        angle = math.atan2(dy, dx)

        if action == 0:
            return nx, ny, True, angle

        elif action == 1:
            return -nx, -ny, True, angle

        elif action == 2:
            return -ny, nx, True, angle

        elif action == 3:
            return ny, -nx, True, angle

        elif action == 4:
            return 0, 0, True, angle

        elif action == 5:
            return nx, ny, False, angle

        return 0, 0, False, angle

    def calculate_fitness(self, me):

        reward = 0.0

        reward += 0.02

        hp_loss = self.prev_hp - me["hp"]

        reward -= hp_loss * 0.3

        enemy_hit_gain = (
            me["enemy_hits"] - self.prev_enemy_hits
        )

        player_hit_gain = (
            me["player_hits"] - self.prev_player_hits
        )

        reward += enemy_hit_gain * 3.0

        reward += player_hit_gain * 8.0

        if me["hp"] <= 0:
            reward -= 10

        self.agent.fitness += reward

        self.prev_hp = me["hp"]

        self.prev_enemy_hits = me["enemy_hits"]
        self.prev_player_hits = me["player_hits"]


# =========================================================
# TRAIN LOOP
# =========================================================

async def evaluate_population(trainer):

    for idx, agent in enumerate(trainer.population):

        agent.fitness = 0

        print(
            "EVALUATING AGENT",
            idx,
            "/",
            POPULATION_SIZE
        )

        bot = GeneticBot(agent)

        await bot.run()


async def main():

    trainer = GeneticTrainer()

    while True:

        await evaluate_population(trainer)

        trainer.evolve()


if __name__ == "__main__":

    asyncio.run(main())
