# main.py
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from game import Game, PROTOCOL_RAYCAST

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("hacker2")

# Глобальные переменные для состояния
game = Game()
client_inputs = {}  # {client_id: input_data}
active_connections = {}  # {client_id: WebSocket}
spectator_connections = {}  # {spectator_id: WebSocket} # Для зрителей

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Современный способ управления жизненным циклом приложения.
    Запускает игровой цикл при старте и корректно останавливает его.
    """
    logger.info("🚀 Starting Game Loop background task...")
    # Запускаем игровой цикл в фоне
    task = asyncio.create_task(game_loop())
    yield  # Приложение работает здесь
    # Остановка (если нужно)
    task.cancel()
    logger.info("🛑 Game Loop stopped.")

app = FastAPI(title="Hacker 2.0", lifespan=lifespan)

# Монтируем статику (наш фронтенд)
app.mount("/static", StaticFiles(directory="public"), name="static")

async def game_loop():

    """
    Turbo RL game loop.

    Features:
    - realtime mode when humans connected
    - turbo simulation when only bots
    - no websocket spam in training mode
    - accelerated evolution/RL training
    """

    while True:

        try:

            # =====================================================
            # INPUTS
            # =====================================================

            for cid, inp in list(client_inputs.items()):

                if cid in active_connections:

                    # game.process_inputs(
                    #     cid,
                    #     inp.get("dx", 0),
                    #     inp.get("dy", 0),
                    #     inp.get("shoot", False),
                    #     inp.get("angle", 0)
                    # )
                    game.process_inputs(
                        cid=cid,
                        move_left_right=inp.get("move_left_right", 0),
                        move_front_back=inp.get("move_front_back", 0),
                        shoot=inp.get("shoot", False),
                        look_angle=inp.get("angle", 0)
                    )

                client_inputs[cid] = {
                    "dx": 0,
                    "dy": 0,
                    "shoot": False
                }

            # =====================================================
            # DETECT REAL HUMANS
            # =====================================================

            real_players = 0

            for p in game.players.values():

                if not p.is_bot:
                    real_players += 1

            has_spectators = len(spectator_connections) > 0

            realtime_mode = (
                real_players > 0
                or has_spectators
            )

            # =====================================================
            # FAST MODE
            # =====================================================
            
            if realtime_mode:

                game.set_fast_mode(False)

            else:

                game.set_fast_mode(True)

            # =====================================================
            # SIMULATION
            # =====================================================

            # for _ in range(game.simulation_speed):
            if True:
                game.tick()

            # =====================================================
            # SNAPSHOT
            # =====================================================

            snapshot_cache = {}
                # snapshot = json.dumps(
                #     game.get_snapshot()
                # )
            
            # Формируем данные для игроков
            for cid in active_connections:
                player = game.players.get(cid)
                if not player:
                    continue
                # snapshot_cache[cid] = json.dumps(
                #     game.get_snapshot_for(player)
                # )
                if player.protocol_version >= PROTOCOL_RAYCAST:
                    # Предполагается что с 2 протоколом и выше только боты
                    snapshot = game.get_bot_observation(
                        player
                    )
                    snapshot_cache[cid] = json.dumps(snapshot)
                else:
                    if realtime_mode:
                        # Людям отправляем только если включен режим realtime_mode
                        snapshot = game.get_snapshot_for(
                            player
                        )
                    snapshot_cache[cid] = json.dumps(snapshot)

            dead_clients = []

            # =================================================
            # PLAYERS
            # =================================================

            for sid, ws in list(active_connections.items()):

                try:

                    # await ws.send_text(snapshot)
                    await asyncio.wait_for(
                        # ws.send_text(snapshot),
                        ws.send_text(snapshot_cache.get(sid, "{}")),
                        timeout=0.01
                    )

                except Exception as e:

                    logger.warning(
                        f"❌ Failed to send to {sid}: {e}"
                    )

                    dead_clients.append(sid)

            # =================================================
            # SPECTATORS
            # =================================================

            for sid, ws in list(spectator_connections.items()):

                try:

                    # await ws.send_text(snapshot)
                    # await asyncio.wait_for(
                    #     # ws.send_text(snapshot),
                    #     ws.send_text(snapshot_cache.get(sid, "{}")),
                    #     timeout=0.01
                    # )

                    # Для зрителей всегда отправляем полный снимок от первого игрока (или пустой, если нет игроков)
                    target_player = next(
                        iter(game.players.values()),
                        None
                    )

                    snapshot = json.dumps(
                        game.get_snapshot_for(target_player)
                    )

                    await asyncio.wait_for(
                        ws.send_text(snapshot),
                        timeout=0.01
                    )


                except Exception:

                    spectator_connections.pop(
                        sid,
                        None
                    )

            # =================================================
            # CLEANUP
            # =================================================

            for sid in dead_clients:

                active_connections.pop(sid, None)

                client_inputs.pop(sid, None)

                game.remove_player(sid)

                logger.info(
                    f"🔌 Cleaned up disconnected client {sid} (total players: {len(active_connections)})"
                )

            # Задержка для реального времени или ускорение для тренировки
            if realtime_mode:
                # realtime FPS
                await asyncio.sleep(1 / 60)

            else:

                # =================================================
                # TURBO TRAINING MODE
                # =================================================

                # no websocket
                # no rendering
                # no sleeps

                await asyncio.sleep(0)

        except Exception as e:

            logger.error(
                f"💥 Critical error in game loop: {e}",
                exc_info=True
            )

            await asyncio.sleep(1)

# --- Эндпоинты ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Принимаем соединение
    await ws.accept()
    client_id = id(ws)
    
    # Регистрируем клиента
    active_connections[client_id] = ws
    client_inputs[client_id] = {"dx": 0, "dy": 0, "shoot": False}
    player = game.add_player(client_id)
    
    logger.info(f"✅ Player {client_id} connected. Total players: {len(active_connections)}")
    
    # Отправляем приветственное сообщение с ID клиента
    await ws.send_text(json.dumps({
        "type": "welcome",
        "player_id": player.id
    }))

    # optional hello packet from bot
    try:
        hello_raw = await asyncio.wait_for(ws.receive_text(), timeout=0.3)
        hello = json.loads(hello_raw)

        if hello.get("type") == "hello":
            bot_name = hello.get("name", "Bot")[:24]
            player.name = bot_name
            player.is_bot = True
            player.protocol_version = hello.get(
                "protocol",
                1
            )

    except Exception:
        pass

    try:
        while True:
            # Ждём данные от клиента
            data = await ws.receive_text()
            try:
                # Парсим и сохраняем ввод
                client_inputs[client_id] = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Invalid JSON from {client_id}")
                
    except WebSocketDisconnect:
        logger.info(f"🔌 Player {client_id} disconnected (clean)")
    except Exception as e:
        logger.error(f"⚠️ Error handling {client_id}: {e}")
    except Exception as e:
        logger.exception(f"❌ WS receive error: {e}")
    finally:
        # try:
        #     await ws.close(code=1000)  # 1000 = нормальное закрытие
        # except:
        #     pass  # Игнорируем, если уже закрыт
        # Гарантированная очистка ресурсов при разрыве
        active_connections.pop(client_id, None)
        client_inputs.pop(client_id, None)
        game.remove_player(client_id)
        logger.info(f"🧹 Cleaned up resources for {client_id}")


@app.websocket("/justlook")
async def spectator_endpoint(ws: WebSocket):

    await ws.accept()

    spectator_id = id(ws)

    spectator_connections[spectator_id] = ws

    logger.info(f"👁 Spectator connected {spectator_id}")

    try:
        while True:
            # spectator ничего не отправляет
            await ws.receive_text()

    except WebSocketDisconnect:
        logger.info(f"👁 Spectator disconnected {spectator_id}")

    finally:
        spectator_connections.pop(spectator_id, None)


@app.get("/")
async def serve_frontend():
    """Отдаёт главную страницу игры"""
    return FileResponse("public/index.html")

@app.get("/health")
async def health_check():
    """Простая проверка: жив ли сервер"""
    return {"status": "ok", "players": len(active_connections)}