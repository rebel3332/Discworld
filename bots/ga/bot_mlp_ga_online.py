import asyncio
import json
import math
import random
import websockets
import pickle
import os
import sys
import argparse
from datetime import datetime
from collections import deque

SERVER = "ws://127.0.0.1:8000/ws"
DEFAULT_SAVE_PATH = "best_mlp_genome.pkl"

"""
Запуск нового обучения
python bot_mlp_ga_online.py

Продолжение обучения с сохранённого генома
python bot_mlp_ga_online.py --load best_mlp_genome.pkl

Настройка параметров (например, увеличить популяцию и жизни на геном)
python bot_mlp_ga_online.py --pop 20 --lives 5 --save my_champion.pkl
"""

class GA_MLP_Bot:
    def __init__(self, pop_size=12, lives_per_genome=3, max_ticks_forced_evolve=500, load_path=None):
        self.my_id = None
        self.pop_size = pop_size
        self.lives_per_genome = lives_per_genome
        self.max_ticks_forced_evolve = max_ticks_forced_evolve

        # 🧠 Архитектура MLP: 8 → 16 → 4
        self.input_dim = 8
        self.hidden_dim = 16
        self.output_dim = 4
        self.gene_len = (self.input_dim * self.hidden_dim + self.hidden_dim +
                         self.hidden_dim * self.output_dim + self.output_dim)

        # 🏆 Hall of Fame — абсолютный чемпион (никогда не теряется)
        self.hall_of_fame = None
        self.best_ever_fitness = -999999

        # Загрузка предобученного генома (если указан путь)
        if load_path and os.path.exists(load_path):
            print(f"📥 Loading genome from {load_path}...")
            loaded = self._load_genome_file(load_path)
            if loaded:
                self.hall_of_fame = loaded["genome"]
                self.best_ever_fitness = loaded.get("best_fitness", 0)
                print(f"✅ Loaded! Best fitness: {self.best_ever_fitness:.2f}")
        
        # Инициализация популяции
        if self.hall_of_fame:
            # Если загрузили чемпиона, начинаем с него + случайные мутации вокруг
            self.population = [self.hall_of_fame[:] if i == 0 else 
                              [g + random.gauss(0, 0.5) for g in self.hall_of_fame] 
                              for i in range(self.pop_size)]
        else:
            self.population = [[random.gauss(0, 1) for _ in range(self.gene_len)] for _ in range(self.pop_size)]
        
        self.fitness = [0.0] * self.pop_size
        self.lives_played = [0] * self.pop_size
        self.genome_stats = [{"kills":0, "dmg_dealt":0, "dmg_taken":0, "ticks":0, "fit":0.0} for _ in range(self.pop_size)]

        self.curr_idx = 0
        self.active_genome = self.population[0]
        self._decode_genome()

        # Статистика жизни
        self.is_alive = True
        self.first_spawn = True
        self.ep_hp_loss = 0
        self.ep_score_gain = 0
        self.ep_ticks = 0
        self.ep_kills = 0
        self.prev_hp = 100
        self.prev_score = 0

        # Статистика поколения
        self.generation = 0
        self.gen_history = deque(maxlen=50)

    def _decode_genome(self):
        """Разворачивает плоский геном в матрицы весов и смещений MLP."""
        g = self.active_genome
        idx = 0
        self.W1 = [g[idx + i*self.hidden_dim : idx + (i+1)*self.hidden_dim] for i in range(self.input_dim)]
        idx += self.input_dim * self.hidden_dim
        self.b1 = g[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        self.W2 = [g[idx + i*self.output_dim : idx + (i+1)*self.output_dim] for i in range(self.hidden_dim)]
        idx += self.hidden_dim * self.output_dim
        self.b2 = g[idx : idx + self.output_dim]

    @staticmethod
    def _silu(x): return x / (1.0 + math.exp(-x))
    @staticmethod
    def _tanh(x): return math.tanh(x)
    @staticmethod
    def _sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, x))))

    def forward(self, inputs):
        """MLP Forward Pass: Inputs(8) → Hidden(16, SiLU) → Outputs(4)"""
        h = [self._silu(sum(i*w for i, w in zip(inputs, col)) + bias) 
             for col, bias in zip(zip(*self.W1), self.b1)]
        out = [sum(h_val*w for h_val, w in zip(h, col)) + bias 
               for col, bias in zip(zip(*self.W2), self.b2)]
        return self._tanh(out[0]), self._tanh(out[1]), out[2]*0.3, self._sigmoid(out[3]) > 0.5

    async def run(self):
        async with websockets.connect(SERVER) as ws:
            print("🧬 ONLINE GA-MLP BOT CONNECTED")
            self._print_header()
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("type") == "welcome":
                        self.my_id = data["player_id"]
                        await ws.send(json.dumps({"type": "hello", "name": "MLP-GABot"}))
                        continue
                    await self.step(ws, data)
                except websockets.ConnectionClosed:
                    print("⚠️ Connection lost. Reconnecting in 3s...")
                    await asyncio.sleep(3)
                    break
                except KeyboardInterrupt:
                    print("\n⏹️  Ctrl+C detected. Saving best genome...")
                    self._save_best_genome(DEFAULT_SAVE_PATH)
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

        # 🔍 Детект смерти/респауна
        if self.is_alive and current_hp == 100 and self.prev_hp < 80 and not self.first_spawn:
            self._on_death()
            self._reset_stats()
            self.prev_hp = current_hp
            self.prev_score = me["score"]
        elif self.first_spawn:
            self.first_spawn = False
            self.prev_hp = current_hp
            self.prev_score = me["score"]

        # 💓 Пульс-отчёт
        if self.ep_ticks % 100 == 0 and self.is_alive:
            print(f"❤️ Alive | Ticks: {self.ep_ticks:3d} | Genome #{self.curr_idx} | Fit: {self.fitness[self.curr_idx]:.2f}")

        if not self.is_alive: return

        # Сбор статистики
        self.ep_hp_loss += max(0, self.prev_hp - current_hp)
        self.ep_score_gain += max(0, me["score"] - self.prev_score)
        score_diff = me["score"] - self.prev_score
        if score_diff >= 50: self.ep_kills += 1
        elif score_diff >= 20: self.ep_kills += 1

        self.prev_hp = current_hp
        self.prev_score = me["score"]

        # 🧠 MLP Inference
        features = self._extract_features(me, enemies, players, bullets)
        move_x, move_y, angle_adj, shoot = self.forward(features)

        targets = enemies + [p for p in players if p["id"] != self.my_id]
        target = min(targets, key=lambda t: math.hypot(t["x"]-me["x"], t["y"]-me["y"]), default=None)
        
        angle = me.get("angle", 0)
        if target:
            dx = target["x"] - me["x"]
            dy = target["y"] - me["y"]
            angle = math.atan2(dy, dx) + angle_adj

        await ws.send(json.dumps({
            "dx": move_x, "dy": move_y, "shoot": shoot, "angle": angle
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
        self.ep_hp_loss = self.ep_score_gain = self.ep_ticks = self.ep_kills = 0

    def _on_death(self):
        fitness = self.ep_score_gain * 1.2 - self.ep_hp_loss * 0.8 + self.ep_ticks * 0.05 + self.ep_kills * 8.0
        self.fitness[self.curr_idx] += fitness
        self.lives_played[self.curr_idx] += 1
        self.genome_stats[self.curr_idx]["kills"] += self.ep_kills
        self.genome_stats[self.curr_idx]["dmg_dealt"] += self.ep_score_gain
        self.genome_stats[self.curr_idx]["dmg_taken"] += self.ep_hp_loss
        self.genome_stats[self.curr_idx]["ticks"] += self.ep_ticks
        self.genome_stats[self.curr_idx]["fit"] += fitness

        print(f"💀 [{self.curr_idx:2d}] Life #{self.lives_played[self.curr_idx]} | K:{self.ep_kills:2d} Dmg:+{self.ep_score_gain:.0f}/-{self.ep_hp_loss:.0f} T:{self.ep_ticks:3d} | F:{fitness:5.2f}")

        if self.lives_played[self.curr_idx] >= self.lives_per_genome or self.ep_ticks > self.max_ticks_forced_evolve:
            self._evolve()
            self.lives_played[self.curr_idx] = 0
            self.fitness[self.curr_idx] = 0

        self.curr_idx = max(range(self.pop_size), key=lambda i: self.fitness[i])
        self.active_genome = self.population[self.curr_idx]
        self._decode_genome()
        
        # 🏆 Обновление рекорда + автосохранение
        if self.fitness[self.curr_idx] > self.best_ever_fitness:
            self.best_ever_fitness = self.fitness[self.curr_idx]
            self.hall_of_fame = self.active_genome[:]
            print(f"🏆 NEW BEST! Fitness: {self.best_ever_fitness:.2f} — Auto-saving...")
            self._save_best_genome(DEFAULT_SAVE_PATH)

    def _evolve(self):
        sorted_idx = sorted(range(self.pop_size), key=lambda i: self.fitness[i], reverse=True)
        gen_fits = [self.fitness[i] for i in range(self.pop_size)]
        avg_fit = sum(gen_fits) / len(gen_fits)
        variance = sum((x - avg_fit) ** 2 for x in gen_fits) / len(gen_fits)
        std_fit = variance ** 0.5

        # 🧬 Элитизм: топ-2 + чемпион из зала славы
        elites = [self.population[sorted_idx[0]][:], self.population[sorted_idx[1]][:]]
        if self.hall_of_fame and self.hall_of_fame not in elites:
            elites.append(self.hall_of_fame[:])
        
        new_pop = elites[:]
        while len(new_pop) < self.pop_size:
            p1 = self.population[random.choice(sorted_idx[:max(1, self.pop_size//2)])]
            p2 = self.population[random.choice(sorted_idx[:max(1, self.pop_size//2)])]
            child = [p1[i] if random.choice([0,1]) else p2[i] for i in range(self.gene_len)]
            child = [g + random.gauss(0, 0.35) for g in child]
            child = [max(-3.0, min(3.0, w)) for w in child]  # Клиппинг весов
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        self.gen_history.append({"gen": self.generation, "best": max(gen_fits), "avg": avg_fit, "std": std_fit, "time": datetime.now().strftime("%H:%M:%S")})
        self._print_summary(sorted_idx[0], max(gen_fits), avg_fit, std_fit)

    # ─────────────────────────────────────────────────────────────
    # 💾 СОХРАНЕНИЕ / ЗАГРУЗКА ГЕНОМА
    # ─────────────────────────────────────────────────────────────
    
    def _save_best_genome(self, path):
        """Сохраняет лучшего чемпиона + метаданные."""
        if not self.hall_of_fame:
            print("⚠️ Nothing to save (no best genome yet)")
            return
        try:
            data = {
                "genome": self.hall_of_fame,
                "best_fitness": self.best_ever_fitness,
                "architecture": f"{self.input_dim}→{self.hidden_dim}(SiLU)→{self.output_dim}",
                "gene_len": self.gene_len,
                "saved_at": datetime.now().isoformat(),
                "generation": self.generation
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"💾 Saved best genome to {path} (Fit: {self.best_ever_fitness:.2f})")
        except Exception as e:
            print(f"❌ Save error: {e}")

    def _load_genome_file(self, path):
        """Загружает геном из файла. Возвращает dict или None при ошибке."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Валидация
            if data.get("gene_len") != self.gene_len:
                print(f"⚠️ Genome mismatch: expected {self.gene_len}, got {data.get('gene_len')}")
                return None
            print(f"📦 Loaded: Gen#{data.get('generation', '?')} | Fit:{data.get('best_fitness', 0):.2f}")
            return data
        except FileNotFoundError:
            print(f"⚠️ File not found: {path}")
            return None
        except Exception as e:
            print(f"❌ Load error: {e}")
            return None

    def _print_summary(self, best_idx, best_fit, avg_fit, std_fit):
        print("\n" + "═" * 70)
        print(f"🧬 GEN #{self.generation} | Best: #{best_idx} ({best_fit:.2f}) | Avg: {avg_fit:.2f}±{std_fit:.2f}")
        print(f"🏆 Best Ever: {self.best_ever_fitness:.2f} | Architecture: 8→16(SiLU)→4")
        if self.hall_of_fame:
            print(f"💾 Hall of Fame: {DEFAULT_SAVE_PATH}")
        print("═" * 70 + "\n")

    def _print_header(self):
        print("\n" + "🎮" * 30)
        print("   🧬 ONLINE GA-MLP BOT v3.1")
        print("   8→16(SiLU)→4 • Real-time • Save/Load • Pure Python")
        print("🎮" * 30 + "\n")
        print(f"📦 Pop: {self.pop_size} | Lives/Genome: {self.lives_per_genome} | Genes: {self.gene_len}")
        if self.hall_of_fame:
            print(f"🏆 Pre-loaded champion | Best fitness: {self.best_ever_fitness:.2f}")
        print("─" * 70 + "\n")

async def main():
    # 🎛️ Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="GA-MLP Bot with Save/Load")
    parser.add_argument("--load", type=str, default=None, help="Path to pre-trained genome (.pkl)")
    parser.add_argument("--save", type=str, default=DEFAULT_SAVE_PATH, help="Path to save best genome")
    parser.add_argument("--pop", type=int, default=12, help="Population size")
    parser.add_argument("--lives", type=int, default=3, help="Lives per genome evaluation")
    args = parser.parse_args()

    bot = GA_MLP_Bot(
        pop_size=args.pop,
        lives_per_genome=args.lives,
        max_ticks_forced_evolve=450,
        load_path=args.load
    )
    
    try:
        await bot.run()
    finally:
        # Гарантированное сохранение при любом выходе
        if bot.hall_of_fame:
            bot._save_best_genome(args.save)

if __name__ == "__main__":
    asyncio.run(main())