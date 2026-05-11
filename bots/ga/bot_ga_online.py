import asyncio
import json
import math
import random
import websockets
from collections import deque
import statistics
from datetime import datetime

SERVER = "ws://127.0.0.1:8000/ws"

class OnlineGABot:
    """
    Бот с Online Genetic Algorithm.
    - Играет наравне с другими в реальном времени.
    - Сам собирает статистику (выживание, урон, очки).
    - Эволюционирует после каждой смерти (смена поколения).
    - Всегда использует лучший найденный геном.


Как это работает в реальном времени
Встроенная популяция: Бот хранит в памяти pop_size геномов (наборов весов).
Онлайн-оценка: Каждый геном тестируется в течение lives_per_genome жизней. Бот сам считает:
ep_score_gain (попадания/убийства)
ep_hp_loss (полученный урон)
ep_ticks (время выживания)
Эволюция по смерти: Как только бот умирает, считается fitness. Если текущий геном "отыграл" достаточно жизней, запускается _evolve():
Отбираются лучшие родители.
Создаются дети через скрещивание + мутацию.
Худшие гены заменяются.
Элитизм: лучший геном всегда сохраняется и сразу берется в игру.
Непрерывная игра: Бот никогда не останавливается. Он просто меняет "мозги" (веса) на лету, постепенно становясь агрессивнее, осторожнее или точнее в зависимости от того, что приносит больше очков в вашей конкретной партии.


    """
    def __init__(self, pop_size=10, lives_per_genome=3, max_ticks_forced_evolve=600):
        self.my_id = None
        self.pop_size = pop_size
        self.lives_per_genome = lives_per_genome
        self.max_ticks_forced_evolve = max_ticks_forced_evolve
        self.input_dim = 8
        self.output_dim = 4
        self.gene_len = self.input_dim * self.output_dim

        self.population = [[random.gauss(0, 1) for _ in range(self.gene_len)] for _ in range(self.pop_size)]
        self.fitness = [0.0] * self.pop_size
        self.lives_played = [0] * self.pop_size
        self.genome_stats = [{"kills": 0, "damage_dealt": 0, "damage_taken": 0, "survival_ticks": 0, "total_fitness": 0.0} for _ in range(self.pop_size)]

        self.curr_idx = 0
        self.active_genome = self.population[0]
        self.W = [self.active_genome[i * self.output_dim : (i + 1) * self.output_dim] for i in range(self.input_dim)]

        self.is_alive = True
        self.first_spawn = True
        self.ep_hp_loss = 0
        self.ep_score_gain = 0
        self.ep_ticks = 0
        self.ep_kills = 0
        self.prev_hp = 100
        self.prev_score = 0

        self.generation = 0
        self.gen_history = deque(maxlen=50)
        self.best_ever_fitness = -999999

    async def run(self):
        async with websockets.connect(SERVER) as ws:
            print("🧬 ONLINE GA BOT CONNECTED")
            self._print_header()
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("type") == "welcome":
                        self.my_id = data["player_id"]
                        await ws.send(json.dumps({"type": "hello", "name": "GABot"}))
                        continue
                    await self.step(ws, data)
                except websockets.ConnectionClosed:
                    print("⚠️ Connection lost. Reconnecting in 3s...")
                    await asyncio.sleep(3)
                    break
                except Exception as e:
                    print(f"❌ BOT ERROR: {e}")
                    break

    async def step(self, ws, state):
        players = state.get("players", [])
        enemies = state.get("enemies", [])
        bullets = state.get("bullets", [])

        me = next((p for p in players if p["id"] == self.my_id), None)
        if not me: return

        current_hp = me["hp"]
        self.ep_ticks += 1

        # 🔍 Умный детект смерти/респауна (сервер мгновенно ставит hp=100)
        if self.is_alive and current_hp == 100 and self.prev_hp < 80 and not self.first_spawn:
            self._on_death()
            self._reset_stats()
            self.prev_hp = current_hp
            self.prev_score = me["score"]
        elif self.first_spawn:
            self.first_spawn = False
            self.prev_hp = current_hp
            self.prev_score = me["score"]

        # Пульс-отчёт каждые 100 тиков
        if self.ep_ticks % 100 == 0 and self.is_alive:
            print(f"❤️ Alive | Ticks: {self.ep_ticks:3d} | Genome #{self.curr_idx} | Fit: {self.fitness[self.curr_idx]:.2f}")

        if not self.is_alive: return

        # Сбор статистики жизни
        self.ep_hp_loss += max(0, self.prev_hp - current_hp)
        self.ep_score_gain += max(0, me["score"] - self.prev_score)
        # Простой детект убийств: +10/20 за выстрел, +50 за фраг
        score_diff = me["score"] - self.prev_score
        if score_diff >= 50: self.ep_kills += 1
        elif score_diff >= 20: self.ep_kills += 1

        self.prev_hp = current_hp
        self.prev_score = me["score"]

        # 🧠 Принятие решения
        print(f"self.W: {self.W}")
        # self.W: [[-0.2538452804241574, 1.622876924727349, 0.25047315205392695, -0.05626491812960774], [-1.078261612290372, -1.349064919355838, -0.27806955136403283, -0.4661857674899596], [1.0008882846400704, 0.30029255840843283, -1.0956948814348222, 0.7354253682815731], [-0.22598303064889275, 0.32100996472024323, -1.6982869716754736, 0.5152557471410151], [-1.06189871865116, 0.28414934819396215, 1.8752223549228098, 1.177740634468195], [0.6433414845909363, 0.5976461993963252, 0.23188565281117915, 0.9790675131704133], [0.7714075852141137, -1.7801730668180278, -1.4591493166078404, -0.8018744950565783], [0.5602201868668824, 0.2014790238019164, 0.23843627265809703, -0.2764780490259791]]
        features = self._extract_features(me, enemies, players, bullets)
        # print(f"features: {features}")
        # features: [-0.8424767811394548, -0.5387325941983498, 0.0032488714618358378, 0.4, 0.717040756084948, -0.6970310712814423, 1.0, 0.0]
        actions = [sum(f * w for f, w in zip(features, col)) for col in zip(*self.W)]

        move_x = math.tanh(actions[0])
        move_y = math.tanh(actions[1])
        angle_adj = actions[2] * 0.3
        shoot_prob = 1.0 / (1.0 + math.exp(-max(-5, min(5, actions[3]))))

        targets = enemies + [p for p in players if p["id"] != self.my_id]
        target = min(targets, key=lambda t: math.hypot(t["x"]-me["x"], t["y"]-me["y"]), default=None)
        
        angle = me.get("angle", 0)
        if target:
            dx = target["x"] - me["x"]
            dy = target["y"] - me["y"]
            angle = math.atan2(dy, dx) + angle_adj

        await ws.send(json.dumps({
            "dx": move_x, "dy": move_y, "shoot": shoot_prob > 0.5, "angle": angle
        }))

    def _extract_features(self, me, enemies, players, bullets):
        targets = enemies + [p for p in players if p["id"] != self.my_id]
        nearest = min(targets, key=lambda t: math.hypot(t["x"]-me["x"], t["y"]-me["y"]), default=None)

        nx, ny, inv_d = 0.0, 0.0, 0.0
        if nearest:
            dx, dy = nearest["x"]-me["x"], nearest["y"]-me["y"]
            d = math.hypot(dx, dy) + 1e-5
            nx, ny = dx/d, dy/d
            inv_d = 1.0/(1.0+d)

        b_nx, b_ny = 0.0, 0.0
        for b in bullets:
            d = math.hypot(b["x"]-me["x"], b["y"]-me["y"])
            if d < 120:
                b_nx, b_ny = (b["x"]-me["x"])/(d+1e-5), (b["y"]-me["y"])/(d+1e-5)
                break

        enemies_near = sum(1 for t in targets if math.hypot(t["x"]-me["x"], t["y"]-me["y"]) < 250)
        return [nx, ny, inv_d, me["hp"]/100.0, b_nx, b_ny, 1.0, min(enemies_near/5.0, 1.0)]

    def _reset_stats(self):
        self.ep_hp_loss = 0
        self.ep_score_gain = 0
        self.ep_ticks = 0
        self.ep_kills = 0

    def _on_death(self):
        fitness = self.ep_score_gain * 1.2 - self.ep_hp_loss * 0.8 + self.ep_ticks * 0.05 + self.ep_kills * 8.0
        self.fitness[self.curr_idx] += fitness
        self.lives_played[self.curr_idx] += 1
        self.genome_stats[self.curr_idx]["kills"] += self.ep_kills
        self.genome_stats[self.curr_idx]["damage_dealt"] += self.ep_score_gain
        self.genome_stats[self.curr_idx]["damage_taken"] += self.ep_hp_loss
        self.genome_stats[self.curr_idx]["survival_ticks"] += self.ep_ticks
        self.genome_stats[self.curr_idx]["total_fitness"] += fitness

        print(f"💀 [{self.curr_idx:2d}] Life #{self.lives_played[self.curr_idx]} | K:{self.ep_kills:2d} Dmg:+{self.ep_score_gain:.0f}/-{self.ep_hp_loss:.0f} T:{self.ep_ticks:3d} | F:{fitness:5.2f}")

        # Эволюция по условию или по таймауту выживания
        if self.lives_played[self.curr_idx] >= self.lives_per_genome or self.ep_ticks > self.max_ticks_forced_evolve:
            self._evolve()
            self.lives_played[self.curr_idx] = 0
            self.fitness[self.curr_idx] = 0

        self.curr_idx = max(range(self.pop_size), key=lambda i: self.fitness[i])
        self.active_genome = self.population[self.curr_idx]
        self.W = [self.active_genome[i * self.output_dim : (i + 1) * self.output_dim] for i in range(self.input_dim)]
        if self.fitness[self.curr_idx] > self.best_ever_fitness:
            self.best_ever_fitness = self.fitness[self.curr_idx]

    def _evolve(self):
        sorted_idx = sorted(range(self.pop_size), key=lambda i: self.fitness[i], reverse=True)
        gen_fits = [self.fitness[i] for i in range(self.pop_size)]
        avg, std = statistics.mean(gen_fits), statistics.stdev(gen_fits) if len(gen_fits)>1 else 0

        new_pop = [self.population[sorted_idx[0]][:]]
        for _ in range(self.pop_size - 1):
            p1 = self.population[random.choice(sorted_idx[:max(1, self.pop_size//2)])]
            p2 = self.population[random.choice(sorted_idx[:max(1, self.pop_size//2)])]
            child = [p1[i] if random.choice([0,1]) else p2[i] for i in range(self.gene_len)]
            child = [g + random.gauss(0, 0.25) for g in child]
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        self.gen_history.append({"gen": self.generation, "best": max(gen_fits), "avg": avg, "std": std, "time": datetime.now().strftime("%H:%M:%S")})
        self._print_summary(sorted_idx[0], max(gen_fits), avg, std)

    def _print_summary(self, best_idx, best_fit, avg_fit, std_fit):
        print("\n" + "═" * 70)
        print(f"🧬 GEN #{self.generation} | Best: #{best_idx} ({best_fit:.2f}) | Avg: {avg_fit:.2f}±{std_fit:.2f}")
        print(f"🏆 Best Ever: {self.best_ever_fitness:.2f}")
        print("═" * 70 + "\n")

    def _print_header(self):
        print("\n🎮" * 30)
        print("   🧬 ONLINE GENETIC ALGORITHM BOT v2.1")
        print("   Real-time learning • HP-jump detection • Heartbeat logs")
        print("🎮" * 30 + "\n")
        print(f"📦 Pop: {self.pop_size} | Lives/Genome: {self.lives_per_genome} | Force Evolve Ticks: {self.max_ticks_forced_evolve}")
        print("─" * 70 + "\n")

async def main():
    bot = OnlineGABot(pop_size=10, lives_per_genome=3, max_ticks_forced_evolve=500)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())