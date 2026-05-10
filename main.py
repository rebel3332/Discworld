# main.py
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from game import Game

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("hacker2")

# Глобальные переменные для состояния
game = Game()
client_inputs = {}  # {client_id: input_data}
active_connections = {}  # {client_id: WebSocket}

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
    Главный цикл игры: 60 тиков в секунду.
    1. Считывает ввод от клиентов.
    2. Обновляет состояние игры (физика, спавн).
    3. Рассылает новое состояние всем подключенным.
    """
    while True:
        try:
            # 1. Обработка ввода
            for cid, inp in list(client_inputs.items()):
                if cid in active_connections:  # Проверяем, жив ли клиент
                    game.process_inputs(
                        cid, 
                        inp.get("dx", 0), 
                        inp.get("dy", 0), 
                        inp.get("shoot", False), 
                        inp.get("angle", 0)
                    )
                # Сбрасываем ввод, чтобы не применять его каждый кадр
                client_inputs[cid] = {"dx": 0, "dy": 0, "shoot": False}
            
            # 2. Логика игры
            game.tick()
            
            # 3. Подготовка и отправка пакета
            snapshot = json.dumps(game.get_snapshot())
            
            # Рассылаем всем активным клиентам
            dead_clients = []
            for cid, ws in active_connections.items():
                try:
                    await ws.send_text(snapshot)
                except Exception as e:
                    # Если отправка не удалась, клиент, скорее всего, отключился
                    logger.warning(f"❌ Failed to send to {cid}: {e}")
                    dead_clients.append(cid)
            
            # Чистим список от "мертвых душ"
            for cid in dead_clients:
                active_connections.pop(cid, None)
                client_inputs.pop(cid, None)
                game.remove_player(cid)
                logger.info(f"🔌 Cleaned up disconnected client {cid}")

            # Держим частоту 60 FPS (1/60 секунды)
            await asyncio.sleep(1/60)
            
        except Exception as e:
            logger.error(f"💥 Critical error in game loop: {e}", exc_info=True)
            await asyncio.sleep(1) # Пауза чтобы не спамить ошибками если цикл падает постоянно

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
    finally:
        try:
            await ws.close(code=1000)  # 1000 = нормальное закрытие
        except:
            pass  # Игнорируем, если уже закрыт
        # Гарантированная очистка ресурсов при разрыве
        active_connections.pop(client_id, None)
        client_inputs.pop(client_id, None)
        game.remove_player(client_id)
        logger.info(f"🧹 Cleaned up resources for {client_id}")

@app.get("/")
async def serve_frontend():
    """Отдаёт главную страницу игры"""
    return FileResponse("public/index.html")

@app.get("/health")
async def health_check():
    """Простая проверка: жив ли сервер"""
    return {"status": "ok", "players": len(active_connections)}