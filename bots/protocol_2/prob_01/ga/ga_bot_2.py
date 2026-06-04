# ga_bot_verified.py
# добавил скрытый слой


# Рождение: Берётся текущий чемпион (individuals[0]), 
# от него создаётся мутированная копия (child).
# 🎮 Сразу в игру: Этот child мгновенно становится активным ботом. 
# Метод get_next_genome() возвращает именно его, и он подключается к серверу.
# Оценка: Только когда бот умирает, скрипт вычисляет его fitness (score) и 
# вызывает report_life_end.
# 🔄 Сортировка: Вся популяция (включая этого новичка) сортируется по очкам.
# ⚖️ Что происходит с «не лучшим» геномом?
# Если сыграл средне/плохо: Его очки окажутся в середине или внизу списка. 
# Он останется в популяции, но в следующем цикле снова окажется в 
# слоте [-1] (худший) и будет заменён новой мутацией от текущего чемпиона.
# Если сыграл хорошо: Его очки высокие → он поднимается в рейтинге. 
# Может стать 2-м, 3-м или даже новым чемпионом. В будущих циклах именно 
# от него могут рождаться новые мутации.




# ga_bot_mlp.py
import asyncio
import json
import math
import random
import numpy as np
import websockets
import time

SERVER = "ws://localhost:8000/ws"
PROTOCOL_VERSION = 2

# =========================================================
# GENOME (MLP: 7 → 10 → 4)
# =========================================================

class Genome:
    def __init__(self, in_dim=7, hidden_dim=10, out_dim=4, weights=None):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fitness = 0
        
        if weights is not None:
            self.w1, self.w2 = weights
        else:
            # Инициализация весов (подходит для tanh)
            limit1 = np.sqrt(6.0 / (in_dim + hidden_dim))
            limit2 = np.sqrt(6.0 / (hidden_dim + 1 + out_dim))
            self.w1 = np.random.uniform(-limit1, limit1, (in_dim, hidden_dim))
            self.w2 = np.random.uniform(-limit2, limit2, (hidden_dim + 1, out_dim))
            
    def forward(self, observation):
        w = observation.get("wall_sensors", [0]*3)
        e = observation.get("enemy_sensors", [0]*3)
        
        # Нормализация входов + bias
        inputs = np.array(
            [min(max(x, 0.0), 2.0) for x in w] + 
            [min(max(x, 0.0), 2.0) for x in e] + 
            [1.0]
        )
        
        # Слой 1: Вход → Скрытый (10 нейронов)
        hidden = np.dot(inputs, self.w1)
        hidden_act = np.tanh(hidden)
        
        # Добавляем bias к скрытому слою
        hidden_with_bias = np.concatenate([hidden_act, [1.0]])
        
        # Слой 2: Скрытый → Выход (4 действия)
        outputs = np.dot(hidden_with_bias, self.w2)
        # print(f"🧠 MLP Output: {outputs}")
        
        # Декодирование выходов
        return {
            "dx": float(np.tanh(outputs[0])),
            "dy": float(np.tanh(outputs[1])),
            "angle_delta": float(np.tanh(outputs[2]) * math.pi * 0.25),
            "shoot": bool(outputs[3] > 0.0)
        }

    def mutate(self, rate=0.1, strength=0.1):
        # Мутация первого слоя
        mask1 = np.random.rand(*self.w1.shape) < rate
        self.w1 += mask1 * np.random.randn(*self.w1.shape) * strength
        
        # Мутация второго слоя
        mask2 = np.random.rand(*self.w2.shape) < rate
        self.w2 += mask2 * np.random.randn(*self.w2.shape) * strength
        return self

    def save(self, path):
        # Явно создаём массив объектов, чтобы избежать ошибки inhomogeneous shape
        np.save(path, np.array([self.w1, self.w2], dtype=object))
        print(f"💾 Saved: {path}")

    @staticmethod

    def load(path):
        w1, w2 = np.load(path, allow_pickle=True)
        return Genome(weights=(w1, w2))


# =========================================================
# POPULATION
# =========================================================

class Population:
    def __init__(self, size=10):
        self.size = size
        self.individuals = [Genome() for _ in range(size)]
        self.generation = 1
        self.best_ever = -1
        
    def report_life_end(self, genome, fitness, duration):
        genome.fitness = fitness
        scores = [g.fitness for g in self.individuals]
        best, avg, worst = max(scores), sum(scores)/len(scores), min(scores)
        
        print(f"📊 Gen {self.generation:3d} | Best: {best:5.1f} | Avg: {avg:5.1f} | Worst: {worst:5.1f} | Time: {duration:.1f}s")
        
        if fitness > self.best_ever:
            self.best_ever = fitness
            print(f" NEW RECORD! {fitness:.1f}")
            genome.save("ga_bot_2_best_genome.npy")
            
        self.individuals.sort(key=lambda g: g.fitness, reverse=True)
        
        # Элитаризм: лучший не трогается, мутируем копию
        champion = self.individuals[0]
        child = Genome(weights=(champion.w1.copy(), champion.w2.copy())).mutate(rate=0.15, strength=0.15)
        self.individuals[-1] = child
        self.generation += 1
        
    def get_next_genome(self):
        return self.individuals[-1]


# =========================================================
# BOT CONTROLLER
# =========================================================

class GeneticBot:
    def __init__(self, population):
        self.pop = population
        self.genome = population.get_next_genome()
        self.angle = random.uniform(-math.pi, math.pi)
        self.life_start_time = 0
        self.life_active = False
        self.prev_hp, self.prev_x, self.prev_y = 1.0, 0.0, 0.0
        self.last_score = 0
        
    async def run_forever(self):
        async with websockets.connect(SERVER) as ws:
            await ws.send(json.dumps({"type": "hello", "name": "GA_MLP", "protocol": PROTOCOL_VERSION}))
            print("🟢 Connected. MLP Evolution started...")
            
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("type") != "bot_observation_v1": continue
                    
                self_data = data['self']
                hp, x, y, score = self_data['hp'], self_data['x'], self_data['y'], self_data.get('score', 0)
                
                respawn = (self.prev_hp < 0.3 and hp > 0.8) or (math.hypot(x-self.prev_x, y-self.prev_y) > 30)
                respawn = respawn and self.life_active
                
                if respawn:
                    duration = time.time() - self.life_start_time
                    fitness = self.last_score #+ (duration * 2.0)
                    self.pop.report_life_end(self.genome, fitness, duration)
                    
                    self.genome = self.pop.get_next_genome()
                    self.life_start_time = time.time()
                    self.angle = random.uniform(-math.pi, math.pi)
                elif not self.life_active:
                    self.life_start_time = time.time()
                    self.life_active = True
                    
                self.prev_hp, self.prev_x, self.prev_y, self.last_score = hp, x, y, score
                
                action = self.genome.forward(data)
                self.angle += action['angle_delta']
                self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))
                await ws.send(json.dumps({"dx": action['dx'], "dy": action['dy'], "angle": self.angle, "shoot": action['shoot']}))

async def main():
    pop = Population(size=10)
    try:
        pop.individuals[0] = Genome.load("best_genome.npy")
        print("📦 Loaded champion MLP")
    except: print("🌱 Fresh MLP start")
    
    await GeneticBot(pop).run_forever()

if __name__ == "__main__":
    asyncio.run(main())