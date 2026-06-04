# ga_bot_verified.py


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




import asyncio
import json
import math
import random
import numpy as np
import websockets
import time
import hashlib

SERVER = "ws://localhost:8000/ws"
PROTOCOL_VERSION = 2

class Genome:
    def __init__(self, input_dim=7, output_dim=4, weights=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fitness = 0
        
        if weights is not None:
            self.weights = weights.copy()
        else:
            self.weights = np.random.randn(input_dim, output_dim) * 0.3
            
    def forward(self, observation):
        w = observation.get("wall_sensors", [0]*3)
        e = observation.get("enemy_sensors", [0]*3)
        inputs = np.array([min(x, 2.0) for x in w] + [min(x, 2.0) for x in e] + [1.0])
        out = np.dot(inputs, self.weights)
        return {
            "dx": float(np.tanh(out[0])),
            "dy": float(np.tanh(out[1])),
            "angle_delta": float(np.tanh(out[2]) * math.pi * 0.25),
            "shoot": bool(out[3] > 0.5)
        }

    def mutate(self, rate=0.1, strength=0.1):
        mask = np.random.rand(*self.weights.shape) < rate
        self.weights += mask * np.random.randn(*self.weights.shape) * strength
        return self

    def fingerprint(self):
        # Короткий хеш весов для отслеживания наследования
        return hashlib.md5(self.weights.tobytes()).hexdigest()[:6]

    def save(self, path):
        np.save(path, self.weights)
        print(f"💾 Saved: {path}")

    @staticmethod
    def load(path):
        return Genome(weights=np.load(path))


class Population:
    def __init__(self, size=10):
        self.size = size
        self.individuals = [Genome() for _ in range(size)]
        self.generation = 1
        self.best_ever = -1
        
    def report_life_end(self, genome, fitness, duration):
        genome.fitness = fitness
        
        # Статистика популяции
        scores = [g.fitness for g in self.individuals]
        best, avg, worst = max(scores), sum(scores)/len(scores), min(scores)
        
        print(f"📊 Gen {self.generation:3d} | Best: {best:5.1f} | Avg: {avg:5.1f} | Worst: {worst:5.1f} | Time: {duration:.1f}s")
        
        if fitness > self.best_ever:
            self.best_ever = fitness
            print(f"🏆 NEW RECORD! {fitness:.1f} | Champion FP: {genome.fingerprint()}")
            genome.save("best_genome.npy")
            
        # Сортировка: лучшие слева
        self.individuals.sort(key=lambda g: g.fitness, reverse=True)
        
        # Элитаризм: лучший остается нетронутым
        champion = self.individuals[0]
        
        # Создаем потомка от лучшего + небольшая мутация
        child = Genome(weights=champion.weights.copy()).mutate(rate=0.15, strength=0.15)
        
        # Заменяем худшего
        self.individuals[-1] = child
        self.generation += 1
        
    def get_next_genome(self):
        return self.individuals[-1]


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
            await ws.send(json.dumps({"type": "hello", "name": "GA_Verified", "protocol": PROTOCOL_VERSION}))
            print("🟢 Connected. Watching evolution...")
            
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
                    # ГИБРИДНЫЙ ФИТНЕС: Score + небольшой бонус за время (чтобы打破僵局 при score=0)
                    fitness = self.last_score + (duration * 2.0)
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
        print(" Loaded champion")
    except: print("🌱 Fresh start")
    
    await GeneticBot(pop).run_forever()

if __name__ == "__main__":
    asyncio.run(main())