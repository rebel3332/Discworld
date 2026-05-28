import asyncio
import json
import math

import websockets


SERVER = "ws://127.0.0.1:8000/ws"


async def bot_loop():

    async with websockets.connect(SERVER) as ws:

        print("CONNECTED")

        # =========================================
        # HELLO PACKET
        # =========================================

        await ws.send(json.dumps({
            "type": "hello",
            "name": "RL_Test_Bot",
            "protocol": 2
        }))

        angle = 0

        while True:

            try:

                msg = await ws.recv()

                data = json.loads(msg)

                # =========================================
                # WELCOME
                # =========================================

                if data.get("type") == "welcome":

                    print("WELCOME:", data)

                    continue

                # =========================================
                # PROTOCOL CHECK
                # =========================================

                if data.get("type") != "bot_observation_v1":

                    print("❌ WRONG PROTOCOL RESPONSE")
                    print(data)

                    continue

                # =========================================
                # OBSERVATION
                # =========================================

                sensors = data.get("sensors", [])

                me = data.get("self", {})

                print("SENSORS:", sensors)

                # rays:
                #
                # sensors[0] = left
                # sensors[1] = front
                # sensors[2] = right

                left = sensors[0]
                front = sensors[1]
                right = sensors[2]

                # =========================================
                # SIMPLE AI
                # =========================================

                move_x = 0
                move_y = 0

                # obstacle ahead
                if front < 0.2:

                    # turn toward freer side
                    if left > right:
                        angle -= 0.3
                    else:
                        angle += 0.3

                    print("TURN")

                else:

                    move_x = math.cos(angle)
                    move_y = math.sin(angle)

                    print("FORWARD")

                # =========================================
                # SEND ACTION
                # =========================================

                action = {
                    "dx": move_x,
                    "dy": move_y,
                    "shoot": False,
                    "angle": angle
                }

                await ws.send(
                    json.dumps(action)
                )

                await asyncio.sleep(1 / 60)

            except Exception as e:

                print("ERROR:", e)

                break


asyncio.run(bot_loop())