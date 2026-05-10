"""
Простой бот для тестирования сервера (дружебный). 
Он подключается, получает своё ID, и в каждом кадре ищет ближайшего врага. 
Если враг далеко - приближается, если близко - отдаляется. 
И всегда стреляет в него.
"""

import asyncio
import json
import math

import websockets


SERVER = "ws://127.0.0.1:8000/ws"


async def bot_loop():

    my_id = None

    async with websockets.connect(SERVER) as ws:

        print("BOT CONNECTED")

        while True:

            try:

                msg = await ws.recv()
                data = json.loads(msg)

                # welcome packet
                if data.get("type") == "welcome":
                    my_id = data["player_id"]
                    print("MY ID:", my_id)
                    continue

                state = data

                players = state.get("players", [])
                enemies = state.get("enemies", [])

                me = None

                for p in players:
                    if p["id"] == my_id:
                        me = p
                        break

                if not me:
                    print("I AM DEAD")
                    return

                nearest = None
                nearest_dist = 999999

                # Ищем ближайшего врага для атаки
                for e in enemies:

                    dx = e["x"] - me["x"]
                    dy = e["y"] - me["y"]

                    dist = math.hypot(dx, dy)

                    if dist < nearest_dist:
                        nearest = e
                        nearest_dist = dist
                
                # Ищем блиайшего игрока для атаки
                for p in players:

                    if p["id"] == my_id:
                        continue

                    dx = p["x"] - me["x"]
                    dy = p["y"] - me["y"]

                    dist = math.hypot(dx, dy)

                    if dist < nearest_dist:
                        nearest = p
                        nearest_dist = dist

                move_x = 0
                move_y = 0
                shoot = False
                angle = me.get("angle", 0)

                if nearest:

                    dx = nearest["x"] - me["x"]
                    dy = nearest["y"] - me["y"]

                    dist = math.hypot(dx, dy)

                    angle = math.atan2(dy, dx)

                    if dist > 150:
                        move_x = dx / dist
                        move_y = dy / dist

                    elif dist < 80:
                        move_x = -dx / dist
                        move_y = -dy / dist

                    shoot = True

                action = {
                    "dx": move_x,
                    "dy": move_y,
                    "shoot": shoot,
                    "angle": angle
                }

                await ws.send(json.dumps(action))

                await asyncio.sleep(1/60)

            except Exception as e:
                print("BOT ERROR:", e)
                break


asyncio.run(bot_loop())