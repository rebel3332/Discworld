# game.py
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Entity:
    id: int
    x: float
    y: float
    radius: float

@dataclass
class Player(Entity):
    name: str = "Player"
    hp: int = 100
    score: int = 0
    angle: float = 0.0
    is_bot: bool = False
    shoot_cooldown: float = 0.0 # время до следующего выстрела
    # добавил для bot_ppo_2
    enemy_hits: int = 0
    player_hits: int = 0
    survival_ticks: int = 0

    def respawn(self, width, height):
        self.hp = 100
        self.score = 0
        self.x = random.randint(100, width - 100)
        self.y = random.randint(100, height - 100)

@dataclass
class Enemy(Entity):
    hp: int = 3
    speed: float = 3.5
    model: str = "simple" # для будущих типов врагов

@dataclass
class Bullet(Entity):
    vx: float = 0.0
    vy: float = 0.0
    lifetime: float = 1.5


@dataclass
class HitEffect:
    """Визуальный эффект попадания (только для клиента)"""
    id: int
    x: float
    y: float
    lifetime: float = 0.3  # секунды
    color: str = "#f88"

@dataclass 
class Explosion:
    """Эффект смерти врага"""
    id: int
    x: float
    y: float
    particles: list  # список {dx, dy, life, color}


class Game:
    def __init__(self, width=800, height=600, tick_rate=60):
        self.W, self.H = width, height
        self.tick_rate = tick_rate
        self.dt = 1.0 / tick_rate
        
        self.players: Dict[int, Player] = {}
        self.enemies: List[Enemy] = []
        self.bullets: List[Bullet] = []
        self.next_id = 1
        self.player_counter = 1
        
        self.spawn_timer = 0.0
        self.spawn_interval = 2.0  # сек
        self.enemy_count = 0 # число заспавненных врагов (для прогрессии сложности)

        self.hit_effects: List[HitEffect] = []
        self.explosions: List[Explosion] = []
        self.next_effect_id = 10000  # отдельный счётчик для эффектов

        # turbo training mode
        self.fast_mode = False
        self.simulation_speed = 1

    def set_fast_mode(self, enabled: bool):
        self.fast_mode = enabled
        if enabled:
            # turbo speed
            self.simulation_speed = 50
        else:
            self.simulation_speed = 1

    def add_player(self, cid: int, name: str | None = None) -> Player:
        p = Player(
            id=self.next_id,
            x=random.randint(100, self.W - 100),
            y=random.randint(100, self.H - 100),
            radius=12,
            name=name or f"Player{self.player_counter}"
        )
        self.next_id += 1
        self.players[cid] = p
        self.player_counter += 1
        return p

    def remove_player(self, cid: int):
        self.players.pop(cid, None)

    def remove_dead_bot(self, cid):
        self.remove_player(cid)

    def process_inputs(self, cid: int, dx: float, dy: float, shoot: bool, angle: float):
        if cid not in self.players: return
        p = self.players[cid]
        p.angle = angle
        
        # 🔒 Движение: нормализуем вектор (анти-спидхак)
        if dx != 0 or dy != 0:
            length = math.hypot(dx, dy)
            speed = 4.0
            p.x += (dx / length) * speed * self.dt * 60
            p.y += (dy / length) * speed * self.dt * 60
            # Ограничение полем
            p.x = max(p.radius, min(self.W - p.radius, p.x))
            p.y = max(p.radius, min(self.H - p.radius, p.y))
            
        # 🔒 Выстрел
        if shoot and p.shoot_cooldown <= 0:
            bullet = Bullet(
                id=self.next_id,
                x=p.x,
                y=p.y,
                radius=4,
                vx=math.cos(angle)*10,
                vy=math.sin(angle)*10
            )

            bullet.owner = cid

            self.bullets.append(bullet)
            p.shoot_cooldown = 0.15  # 150ms между выстрелами
            self.next_id += 1

    def _check_collisions(self):
        # Пуля -> враг
        for b in self.bullets[:]:
            bullet_owner = getattr(b, "owner", None)

            for e in self.enemies[:]:
                if math.hypot(b.x - e.x, b.y - e.y) < (b.radius + e.radius):
                    e.hp -= 1
                    b.lifetime = 0

                    # очки владельцу пули
                    if bullet_owner in self.players:
                        owner_player = self.players[bullet_owner]

                        owner_player.score += 10
                        owner_player.enemy_hits += 1

                    # эффект попадания
                    self.hit_effects.append(HitEffect(
                        id=self.next_effect_id,
                        x=e.x,
                        y=e.y,
                        lifetime=0.2
                    ))
                    self.next_effect_id += 1

                    # смерть врага
                    if e.hp <= 0:
                        self._create_explosion(e.x, e.y)

                        if bullet_owner in self.players:
                            self.players[bullet_owner].score += 50

                    break

        # Пуля -> игрок
        for b in self.bullets[:]:

            owner = getattr(b, "owner", None)

            for cid, p in self.players.items():

                # нельзя попасть в себя
                if cid == owner:
                    continue

                if math.hypot(b.x - p.x, b.y - p.y) < (b.radius + p.radius):

                    p.hp -= 10
                    b.lifetime = 0

                    # очки за попадание
                    if owner in self.players:
                        owner_player = self.players[owner]

                        owner_player.score += 20
                        owner_player.player_hits += 1

                    self.hit_effects.append(HitEffect(
                        id=self.next_effect_id,
                        x=p.x,
                        y=p.y,
                        lifetime=0.2,
                        color="#a0f"
                    ))

                    self.next_effect_id += 1

                    # смерть игрока
                    if p.hp <= 0:
                        # if p.is_bot:
                        #     # dead_bots.append(cid)
                        #     pass
                        # else:
                            # p.hp = 100
                            # p.x = random.randint(100, self.W - 100)
                            # p.y = random.randint(100, self.H - 100)
                            p.respawn(self.W, self.H)

                    break



        # Враг -> игрок
        for e in self.enemies[:]:
            for cid, p in self.players.items():
                if math.hypot(p.x - e.x, p.y - e.y) < (p.radius + e.radius):
                    p.hp -= 15
                    e.hp = 0

                    self._create_explosion(e.x, e.y)

                    if p.hp <= 0:
                        # p.hp = 100
                        # p.score = 0
                        # p.x = self.W // 2
                        # p.y = self.H // 2
                        p.respawn(self.W, self.H)


    def _create_explosion(self, x: float, y: float):
        """Создаёт партиклы взрыва"""
        particles = []
        for _ in range(12):
            angle = random.uniform(0, 6.28)
            speed = random.uniform(1, 4)
            particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": random.uniform(0.3, 0.8),
                "color": random.choice(["#f84", "#fa2", "#ff6"])
            })
        self.explosions.append(Explosion(
            id=self.next_effect_id,
            x=x, y=y,
            particles=particles
        ))
        self.next_effect_id += 1

    def world_tick(self) -> bool:
        # ❗ если никого нет в мире — полностью стопаем логику
        if (
            len(self.players) == 0
        ):
            return  False
        return True

    def tick(self):
        if self.world_tick():
            # cooldowns
            for p in self.players.values():
                p.survival_ticks += 1
                if p.shoot_cooldown > 0:
                    p.shoot_cooldown -= self.dt

            # 1. Спавн врагов (без изменений)
            self.spawn_timer += self.dt
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_timer = 0
                self.enemy_count += 1
                if (self.enemy_count % 10 == 0):
                    enemy_model = "simple_boss"
                    enemy_hp = 30
                else:
                    enemy_model = "simple" # вид врага
                    enemy_hp = 3
            
                side = random.choice(['top','bottom','left','right'])
                if side == 'top': x, y = random.uniform(0, self.W), -30
                elif side == 'bottom': x, y = random.uniform(0, self.W), self.H + 30
                elif side == 'left': x, y = -30, random.uniform(0, self.H)
                else: x, y = self.W + 30, random.uniform(0, self.H)
                # self.enemies.append(Enemy(id=self.next_id, x=x, y=y, radius=14))

                self.enemies.append(Enemy(id=self.next_id, x=x, y=y, radius=14, hp=enemy_hp, model=enemy_model))
                self.next_id += 1

            # 2. Логика врагов
            if self.players:
                for e in self.enemies:

                    nearest = None
                    nearest_dist = 999999

                    for p in self.players.values():
                        dist = math.hypot(p.x - e.x, p.y - e.y)

                        if dist < nearest_dist:
                            nearest = p
                            nearest_dist = dist

                    if nearest:
                        dx = nearest.x - e.x
                        dy = nearest.y - e.y

                        dist = math.hypot(dx, dy)

                        if dist > 0:
                            e.x += (dx / dist) * e.speed * self.dt * 60
                            e.y += (dy / dist) * e.speed * self.dt * 60

            # 3. Пули
            for b in self.bullets:
                b.x += b.vx * self.dt * 60
                b.y += b.vy * self.dt * 60
                b.lifetime -= self.dt

            # 4. Коллизии
            self._check_collisions()

            # 5. Очистка сущностей (классический способ)
            # 🔹 Пули: удаляем, если время вышло или улетели за карту
            self.bullets = [b for b in self.bullets if b.lifetime > 0 and 0 < b.x < self.W and 0 < b.y < self.H]
            
            # 🔹 Враги: удаляем, если здоровье <= 0
            self.enemies = [e for e in self.enemies if e.hp > 0]
            
            # 🔹 Эффекты попаданий: обновляем время жизни и фильтруем
            if not self.fast_mode:
                self._update_effects()

    def _update_effects(self):
        """Обновляет и очищает визуальные эффекты (без моржового оператора)"""
        # 1. Эффекты попаданий (Hit Effects)
        for h in self.hit_effects:
            h.lifetime -= self.dt  # Уменьшаем время жизни
        
        # Оставляем только те, что ещё активны
        self.hit_effects = [h for h in self.hit_effects if h.lifetime > 0]
        
        # 2. Взрывы (Explosions)
        for exp in self.explosions:
            for pt in exp.particles:
                pt["x"] += pt["vx"] * self.dt * 60
                pt["y"] += pt["vy"] * self.dt * 60
                pt["life"] -= self.dt
            # Удаляем умершие частицы внутри взрыва
            exp.particles = [p for p in exp.particles if p["life"] > 0]
        
        # Удаляем пустые взрывы
        self.explosions = [e for e in self.explosions if e.particles]

    
    def get_snapshot(self) -> dict:
        return {
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "x": p.x,
                    "y": p.y,
                    "hp": p.hp,
                    "score": p.score,
                    "radius": p.radius,
                    "angle": p.angle,
                    "enemy_hits": p.enemy_hits,
                    "player_hits": p.player_hits,
                    "survival_ticks": p.survival_ticks
                }
                for p in self.players.values()
            ],
            "enemies": [
                {
                    "model": e.model,
                    "id": e.id, 
                    "x": e.x, 
                    "y": e.y, 
                    "radius": e.radius,
                    "hp": e.hp,
                    "flash": e.hp < 3}
                for e in self.enemies
            ],
            "bullets": [
                {"id": b.id, "x": b.x, "y": b.y, "vx": b.vx, "vy": b.vy, "radius": b.radius}
                for b in self.bullets
            ],
            "hits": [
                {
                    "id": h.id,
                    "x": h.x,
                    "y": h.y,
                    "color": h.color
                }
                for h in self.hit_effects
            ],
            "explosions": [
                {"id": ex.id, "x": ex.x, "y": ex.y, "p": ex.particles} 
                for ex in self.explosions[-20:]   # ⬅️ ограничение количества взрывов в снимке (для оптимизации)
            ],
        }