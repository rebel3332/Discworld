# game.py
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict

from world import World
from config import SENSORS

PROTOCOL_LEGACY = 1
PROTOCOL_RAYCAST = 2


ENEMY_TYPES = {

    "boss": {
        "speed": 1.,
        "hp": 40,
        "damage": 8,
        "radius": 24,
        "color": "#6f6",
    },

    "hunter": {
        "speed": 1.2,
        "hp": 4,
        "damage": 8,
        "radius": 14,
        "color": "#6f6",
    },

    "zombie": {
        "speed": 0.5,
        "hp": 1,
        "damage": 20,
        "radius": 14,
        "color": "#9c6",
    },
}

@dataclass
class Entity:
    id: int
    x: float
    y: float
    radius: float
    angle: float = 0.0

@dataclass
class Player(Entity):
    name: str = "Player"
    hp: int = 100
    score: int = 0
    is_bot: bool = False
    shoot_cooldown: float = 0.0 # время до следующего выстрела
    vx: float = 0.0
    vy: float = 0.0
    # добавил для bot_ppo_2
    enemy_hits: int = 0
    player_hits: int = 0
    survival_ticks: int = 0

    protocol_version: int = 1

    def respawn(self, width, height):
        self.hp = 100
        self.score = 0
        self.x = random.randint(100, width - 100)
        self.y = random.randint(100, height - 100)

@dataclass
class Enemy(Entity):
    name: str = "Hunter"
    hp: int = 3
    speed: float = 3.5
    isMoving: bool = False
    enemy_type: str = "hunter"

    def __init__(self, enemy_type="hunter", **kwds):
        params = ENEMY_TYPES.get(enemy_type, ENEMY_TYPES["hunter"]) # Защита от неизвестного типа

        self.name = enemy_type.capitalize()
        self.hp = params.get("hp", 1)
        self.speed = params.get("speed", 1.0)
        self.enemy_type = enemy_type

        # Если парамтры переданы явно, используем их, иначе - из словаря
        kwds.setdefault(
            "radius",
            params.get("radius", 14)
        )

        super().__init__(**kwds)
        print(f"Spawned enemy {self.name} with HP {self.hp} and speed {self.speed}, radius {self.radius}")


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

        self.world = World(world_width=self.W, world_height=self.H)

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
            radius=14,
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

    # def can_move_to(self, x, y, radius):
    #     """Проверяет, можно ли поместиться в точку (x, y) с данным радиусом, не упираясь в стены<br>     (проверяем 4 угла)"""
    #     return (
    #         self.world.isWalkable(x - radius, y - radius) and
    #         self.world.isWalkable(x + radius, y - radius) and
    #         self.world.isWalkable(x - radius, y + radius) and
    #         self.world.isWalkable(x + radius, y + radius)
    #     )

    def can_move_to(self, x, y, radius):
        """Проверяет, можно ли поместиться в точку (x, y) с данным радиусом, не упираясь в стены<br>     (проверяем все клетки, пересекаемые окружностью)"""
        left = int((x - radius) // self.world.TILE_SIZE)
        right = int((x + radius) // self.world.TILE_SIZE)
        top = int((y - radius) // self.world.TILE_SIZE)
        bottom = int((y + radius) // self.world.TILE_SIZE)
        for ty in range(top, bottom + 1):
            for tx in range(left, right + 1):
                wx = tx * self.world.TILE_SIZE + self.world.TILE_SIZE / 2
                wy = ty * self.world.TILE_SIZE + self.world.TILE_SIZE / 2
                if not self.world.isWalkable(wx, wy):
                    return False
        return True
    
    def resolve_collision(self, entity):
        """Если сущность застряла в стене, пытаемся её вытащить в ближайшую свободную точку"""
        if self.can_move_to(
            entity.x,
            entity.y,
            entity.radius
        ):
            return

        offsets = [

            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),

            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ]

        max_push_distance = 32

        for distance in range(
            1,
            max_push_distance + 1
        ):

            for ox, oy in offsets:

                test_x = entity.x + ox * distance

                test_y = entity.y + oy * distance

                if self.can_move_to(
                    test_x,
                    test_y,
                    entity.radius
                ):

                    entity.x = test_x
                    entity.y = test_y

                    return

    def process_inputs(self, cid: int, dx: float, dy: float, shoot: bool, angle: float):
        if cid not in self.players: return
        p = self.players[cid]
        p.angle = angle
        p.vx = dx
        p.vy = dy
        
        # 🔒 Движение: нормализуем вектор (анти-спидхак)
        if dx != 0 or dy != 0:
            length = math.hypot(dx, dy)
            speed = 4.0
            # p.x += (dx / length) * speed * self.dt * 60
            # p.y += (dy / length) * speed * self.dt * 60
            # # Ограничение полем
            # p.x = max(p.radius, min(self.W - p.radius, p.x))
            # p.y = max(p.radius, min(self.H - p.radius, p.y))

            new_x = p.x + (dx / length) * speed * self.dt * 60
            new_y = p.y + (dy / length) * speed * self.dt * 60
            # Ограничение полем (границами мира)
            new_x = max(p.radius, min(self.W - p.radius, new_x))
            new_y = max(p.radius, min(self.H - p.radius, new_y))

            # Проверка коллизий с миром (с учетом радиуса)
            if self.can_move_to(new_x, p.y, p.radius):
                p.x = new_x

            if self.can_move_to(p.x, new_y, p.radius):
                p.y = new_y

            self.resolve_collision(p)
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

    def cast_ray(
        self,
        x,
        y,
        angle,
        max_distance=300,
        step=4
    ):
        """Отправляет луч из точки (x, y) в направлении angle и возвращает информацию о первом столкновении с стеной"""
        dx = math.cos(angle)
        dy = math.sin(angle)

        distance = 0

        while distance < max_distance:

            px = x + dx * distance
            py = y + dy * distance

            if not self.world.isWalkable(px, py):

                return {
                    "distance": distance / max_distance,
                    "hit_x": px,
                    "hit_y": py
                }

            distance += step

        return {
            "distance": 1.0,
            "hit_x": x + dx * max_distance,
            "hit_y": y + dy * max_distance
        }

    def build_sensor_data(self, player: Player):
        """Строит данные для сенсоров бота на основе его положения и мира"""
        if  not player.is_bot or player.protocol_version < PROTOCOL_RAYCAST:
            return {
                "sensors": None,
                "debug_rays": None
            }
        sensors = []

        debug_rays = []

        sensor_configs = SENSORS.get(
            "rays",
            []
        )

        max_distance = SENSORS.get(
            "max_distance",
            300
        )

        step = SENSORS.get(
            "step",
            4
        )

        for sensor in sensor_configs:

            sensor_angle = sensor.get(
                "angle",
                0
            )

            angle = (
                player.angle
                +
                math.radians(sensor_angle)
            )

            ray = self.cast_ray(
                player.x,
                player.y,
                angle,
                max_distance=max_distance,
                step=step
            )

            sensors.append(
                ray["distance"]
            )

            debug_rays.append({
                "x1": player.x,
                "y1": player.y,
                "x2": ray["hit_x"],
                "y2": ray["hit_y"]
            })

        return {
            "sensors": sensors,
            "debug_rays": debug_rays
        }

    # def get_bot_observation(self, player):
    #     sensors = []
    #     debug_rays = []
    #     sensor_configs = SENSORS.get(
    #         "rays",
    #         []
    #     )
    #     max_distance = SENSORS.get(
    #         "max_distance",
    #         300
    #     )
    #     step = SENSORS.get(
    #         "step",
    #         4
    #     )

    #     for sensor in sensor_configs:
    #         sensor_angle = sensor.get(
    #             "angle",
    #             0
    #         )
    #         angle = (
    #             player.angle
    #             +
    #             math.radians(sensor_angle)
    #         )
    #         ray = self.cast_ray(
    #             player.x,
    #             player.y,
    #             angle,
    #             max_distance=max_distance,
    #             step=step
    #         )
    #         sensors.append(
    #             ray["distance"]
    #         )
    #         debug_rays.append({
    #             "x1": player.x,
    #             "y1": player.y,
    #             "x2": ray["hit_x"],
    #             "y2": ray["hit_y"]
    #         })

    #     return {
    #         "type": "bot_observation_v1",
    #         "self": {
    #             "id": player.id,
    #             "x": player.x,
    #             "y": player.y,
    #             "hp": player.hp / 100.0,
    #             "angle": player.angle
    #         },
    #         "sensors": sensors,
    #         "debug_rays": debug_rays
    #     }

    def get_bot_observation(self, player):
        """"""
        sensor_data = self.build_sensor_data(
            player
        )

        return {

            "type": "bot_observation_v1",

            "self": {
                "id": player.id,
                "x": player.x,
                "y": player.y,
                "hp": player.hp / 100.0,
                "angle": player.angle
            },

            "sensors": sensor_data["sensors"]
        }

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

    def is_visible_to_any_player(
        self,
        x,
        y,
        margin=200
    ):
        """проверяем, видна ли точка хотя бы одному игроку """
        for p in self.players.values():
            screen_w = self.W
            screen_h = self.H
            if (
                abs(x - p.x) < screen_w / 2 + margin
                and
                abs(y - p.y) < screen_h / 2 + margin
            ):
                return True
        return False
    
    def is_far_from_players(
        self,
        x,
        y,
        min_distance=500
    ):
        """проверяем, что точка находится на достаточном расстоянии от всех игроков (для спавна врагов)"""
        for p in self.players.values():

            dist = math.hypot(
                p.x - x,
                p.y - y
            )
            if dist < min_distance:
                return False
        return True

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
                # if (self.enemy_count % 10 == 0):
                #     enemy_model = "simple_boss"
                #     enemy_hp = 30
                # else:
                #     enemy_model = "simple" # вид врага
                #     enemy_hp = 3

                enemy_type = random.choices(
                        ["boss", "hunter", "zombie"],
                        weights=[5, 20, 75]
                    )[0]

                # спавн за пределами карты
                # side = random.choice(['top','bottom','left','right'])
                # if side == 'top': x, y = random.uniform(0, self.W), -30
                # elif side == 'bottom': x, y = random.uniform(0, self.W), self.H + 30
                # elif side == 'left': x, y = -30, random.uniform(0, self.H)
                # else: x, y = self.W + 30, random.uniform(0, self.H)

                # Спавн врагов
                # spawned = False
                # # Спавн врагов за границей видимости игроков
                # attempts = 0
                # max_attempts = 50
                # while not spawned and attempts < max_attempts:
                #     attempts += 1

                #     x = random.uniform(50, self.W - 50)
                #     y = random.uniform(50, self.H - 50)
                #     # проверяем, что точка свободна от стен с учетом радиуса врага  
                #     dummy_enemy = Enemy(
                #         id=0,
                #         x=x,
                #         y=y,
                #         radius=14
                #     )

                #     visible = self.is_visible_to_any_player(x, y)
                #     print(
                #         "spawn check",
                #         visible,
                #         x,
                #         y
                #     )

                #     if (
                #         self.can_move_to(x, y, dummy_enemy.radius)
                #         and
                #         self.is_far_from_players(x, y)
                #     ):
                #         self.enemies.append(
                #             Enemy(
                #                 id=self.next_id,
                #                 x=x,
                #                 y=y,
                #                 radius=14,
                #                 hp=enemy_hp,
                #                 model=enemy_model
                #             )
                #         )
                #         self.next_id += 1
                #         print("Silently spawned enemy at")
                #         spawned = True
                # # спавн в случайной точке карты (но не внутри стен)
                # while not spawned:
                #     x = random.uniform(50, self.W - 50)
                #     y = random.uniform(50, self.H - 50)
                #     # проверяем, что точка свободна от стен с учетом радиуса врага  
                #     dummy_enemy = Enemy(
                #         id=0,
                #         x=x,
                #         y=y,
                #         radius=14
                #     )
                #     #
                #     if self.can_move_to(x, y, dummy_enemy.radius):
                #         self.enemies.append(
                #             Enemy(
                #                 id=self.next_id,
                #                 x=x,
                #                 y=y,
                #                 radius=14,
                #                 hp=enemy_hp,
                #                 model=enemy_model
                #             )
                #         )
                #         self.next_id += 1
                #         print("Randomly spawned enemy at")
                #         spawned = True

                # сначала пытаемся заспавнить врага в идеальной точке (невидимо для игроков), 
                # если не получается — спавним в лучшей из найденных точек, 
                # если и её нет — спавним в рандомной точке карты (но не внутри стен)
                spawned = False
                best_spawn = None
                best_distance = -1

                max_attempts = 50

                for _ in range(max_attempts):
                    x = random.uniform(50, self.W - 50)
                    y = random.uniform(50, self.H - 50)
                    radius = 14
                    if not self.can_move_to(x, y, radius):
                        continue
                    nearest_distance = float("inf")
                    visible = False
                    for p in self.players.values():
                        dist = math.hypot(
                            p.x - x,
                            p.y - y
                        )
                        nearest_distance = min(
                            nearest_distance,
                            dist
                        )
                        if self.is_visible_to_any_player(x, y):
                            visible = True

                    # идеальный спавн
                    if not visible:
                        self.enemies.append(
                            Enemy(
                                id=self.next_id,
                                x=x,
                                y=y,
                                radius=radius,
                                enemy_type=enemy_type,
                            )
                        )
                        self.next_id += 1
                        print("Silently spawned enemy")
                        spawned = True
                        break

                    # запоминаем лучший плохой вариант
                    if nearest_distance > best_distance:
                        best_distance = nearest_distance
                        best_spawn = (x, y)

                # если не получилось заспавнить врага в идеальной точке, 
                # спавним в лучшей из найденных точек
                if not spawned and best_spawn:
                    x, y = best_spawn
                    self.enemies.append(
                        Enemy(
                            id=self.next_id,
                            x=x,
                            y=y,
                            enemy_type=enemy_type,
                        )
                    )
                    self.next_id += 1
                    print(
                        f"Fallback spawn at distance {best_distance:.1f}"
                    )

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
                        if dx != 0 or dy != 0:
                            e.isMoving = True
                        else:   
                            e.isMoving = False

                        # if dist > 0:
                        #     e.x += (dx / dist) * e.speed * self.dt * 60
                        #     e.y += (dy / dist) * e.speed * self.dt * 60
                        #     e.angle = math.atan2(dy, dx)

                        # Простое движение с проверкой коллизий (без застревания в стенах)
                        # if dist > 0:
                        #     move_x = (dx / dist) * e.speed * self.dt * 60
                        #     move_y = (dy / dist) * e.speed * self.dt * 60

                        #     new_x = e.x + move_x
                        #     new_y = e.y + move_y

                        #     if self.can_move_to(new_x, e.y, e.radius):
                        #         e.x = new_x
                        #     if self.can_move_to(e.x, new_y, e.radius):
                        #         e.y = new_y
                        #     self.resolve_collision(e)
                        #     e.angle = math.atan2(dy, dx)

                        if dist > 0:
                            # =====================================
                            # Базовое направление к игроку
                            # =====================================

                            dir_x = dx / dist
                            dir_y = dy / dist

                            # =====================================
                            # Проверяем стену впереди
                            # =====================================

                            look_ahead = 24

                            front_x =  e.x + dir_x * look_ahead

                            front_y = e.y + dir_y * look_ahead

                            blocked = not self.can_move_to(
                                    front_x,
                                    front_y,
                                    e.radius
                                )

                            # =====================================
                            # Obstacle avoidance
                            # =====================================
                            if blocked:
                                # Левый вектор
                                left_x = -dir_y
                                left_y = dir_x
                                # Правый вектор
                                right_x = dir_y
                                right_y = -dir_x
                                left_free = self.can_move_to(
                                        e.x + left_x * look_ahead,
                                        e.y + left_y * look_ahead,
                                        e.radius
                                    )
                                right_free = self.can_move_to(
                                        e.x + right_x * look_ahead,
                                        e.y + right_y * look_ahead,
                                        e.radius
                                    )
                                # Пытаемся обходить препятствие
                                if left_free:
                                    dir_x += left_x * 0.8
                                    dir_y += left_y * 0.8
                                elif right_free:
                                    dir_x += right_x * 0.8
                                    dir_y += right_y * 0.8

                            # =====================================
                            # Нормализация
                            # =====================================
                            length = math.hypot(dir_x, dir_y)
                            if length > 0:
                                dir_x /= length
                                dir_y /= length

                            # =====================================
                            # Финальное движение
                            # =====================================
                            move_x = dir_x * e.speed * self.dt * 60

                            move_y = dir_y * e.speed * self.dt * 60

                            new_x = e.x + move_x
                            new_y = e.y + move_y

                            moved = False

                            if self.can_move_to(
                                new_x,
                                e.y,
                                e.radius
                            ):
                                e.x = new_x
                                moved = True

                            if self.can_move_to(
                                e.x,
                                new_y,
                                e.radius
                            ):
                                e.y = new_y
                                moved = True

                            # Push-out resolve
                            self.resolve_collision(e)

                            e.angle = math.atan2(dir_y, dir_x)

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
        # Получаем игрока (для определения центра мира)
        player = next(
            iter(self.players.values()),
            None
        )
        # Получаем чанки вокруг игрока (если он есть)
        chunks = []
        if player:
            chunks = self.world.get_chunks_around(
                player.x,
                player.y,
                radius=1
            )

        return {
            "world": {
                "chunk_size": self.world.CHUNK_SIZE,
                "tile_size": self.world.TILE_SIZE,
            },
            "chunks": chunks,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "x": p.x,
                    "y": p.y,
                    "vx": p.vx,
                    "vy": p.vy,
                    "hp": p.hp,
                    "score": p.score,
                    "radius": p.radius,
                    "angle": p.angle,
                    "enemy_hits": p.enemy_hits,
                    "player_hits": p.player_hits,
                    "survival_ticks": p.survival_ticks,
                }
                for p in self.players.values()
            ],
            "enemies": [
                {
                    "name": e.name,
                    "id": e.id, 
                    "x": e.x, 
                    "y": e.y, 
                    "angle": e.angle,
                    "radius": e.radius,
                    "hp": e.hp,
                    "enemy_type": e.enemy_type,
                    "isMoving": e.isMoving,
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
    
    def get_snapshot_for(self, player=None) -> dict:

        # spectator mode
        if player is None:

            player = next(
                iter(self.players.values()),
                None
            )

        chunks = []

        if player:

            chunks = self.world.get_chunks_around(
                player.x,
                player.y,
                radius=1
            )

            

        return {

            "world": {
                "chunk_size": self.world.CHUNK_SIZE,
                "tile_size": self.world.TILE_SIZE,
            },

            "chunks": chunks,

            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "x": p.x,
                    "y": p.y,
                    "vx": p.vx,
                    "vy": p.vy,
                    "hp": p.hp,
                    "score": p.score,
                    "radius": p.radius,
                    "angle": p.angle,
                    "is_bot": p.is_bot,
                    "debug_rays": self.build_sensor_data(p)["debug_rays"],
                }
                for p in self.players.values()
            ],

            "enemies": [
                {
                    "name": e.name,
                    "id": e.id,
                    "x": e.x,
                    "y": e.y,
                    "angle": e.angle,
                    "radius": e.radius,
                    "hp": e.hp,
                    "enemy_type": e.enemy_type,
                    "isMoving": e.isMoving,
                }
                for e in self.enemies
            ],

            "bullets": [
                {
                    "id": b.id,
                    "x": b.x,
                    "y": b.y,
                    "vx": b.vx,
                    "vy": b.vy,
                    "radius": b.radius
                }
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
        }