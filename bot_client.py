import asyncio
import json
import math
import random
import websockets

async def run_bot():

    uri = "ws://localhost:8000/bot"

    async with websockets.connect(uri) as ws:

        while True:

            data = json.loads(await ws.recv())

            if data["done"]:
                print("BOT DEAD")
                return

            obs = data["obs"]

            action = {
                "dx": random.uniform(-1, 1),
                "dy": random.uniform(-1, 1),
                "shoot": random.random() > 0.7,
                "angle": random.uniform(-math.pi, math.pi)
            }

            await ws.send(json.dumps(action))

asyncio.run(run_bot())