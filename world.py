# world.py

import random
from config import WORLDS





class World:

    def __init__(self, seed=12345, world_width=40, world_height=40):

        self.seed = seed

        self.world_width = world_width
        self.world_height = world_height

        self.chunks = {}
        self.TILE_GRASS = 0
        self.TILE_STONE = 1
        self.TILE_TOXIC = 2
        self.TILE_WATER = 3
        self.CHUNK_SIZE = 30 # в тайлах (например, 16x16)
        self.TILE_SIZE = 32 # в пикселях

        self.config = WORLDS

        self.biome_name = self.config["active_biome"]
        self.biome = self.config["biomes"][self.biome_name]

    def get_chunk(self, cx, cy):

        key = (cx, cy)

        # if not self.chunk_exists(cx, cy):
        #     return None

        if key not in self.chunks:

            self.chunks[key] = self.generate_chunk(cx, cy)

        return self.chunks[key]

    def generate_chunk(self, cx, cy):
        random.seed(
            self.seed + cx * 9999 + cy
        )
        tiles = []
        for y in range(self.CHUNK_SIZE):
            row = []
            for x in range(self.CHUNK_SIZE):
                world_x = self.TILE_SIZE * (cx * self.CHUNK_SIZE + x)
                world_y = self.TILE_SIZE * (cy * self.CHUNK_SIZE + y)

                if (
                    world_x < 0 or
                    world_y < 0 or
                    world_x >= self.world_width or
                    world_y >= self.world_height
                ):
                    # За пределами мира - вода
                    row.append([
                        self.TILE_WATER,
                        0
                    ])

                else:
                    ## Генерация тайла исходя из захардкоженной вероятности
                    # v = (
                    #     random.random()
                    # )
                    # if v < 0.1:
                    #     ground = self.TILE_STONE
                    #     walkable = 0
                    # elif v < 0.2:
                    #     ground = self.TILE_TOXIC
                    #     walkable = 1
                    # else:
                    #     ground = self.TILE_GRASS
                    #     walkable = 1
                    # row.append([
                    #     ground,
                    #     walkable
                    # ])

                    # Генерация тайла 
                    # Исходя из верояностей поялвения тайлов выбранного биома
                    tile_weights = self.biome["tile_weights"]

                    tile_names = list(tile_weights.keys())
                    weights = list(tile_weights.values())

                    tile_name = random.choices(
                        tile_names,
                        weights=weights
                    )[0]
                    tile_id = self.config["tiles"][tile_name]

                    props = self.config["tile_properties"][tile_name]

                    walkable = 1 if props["walkable"] else 0

                    row.append([
                        tile_id,
                        walkable
                    ])

            tiles.append(row)

        return {
            "cx": cx,
            "cy": cy,
            "tiles": tiles
        }

    def get_tile(self, tx, ty):

        cx = tx // self.CHUNK_SIZE
        cy = ty // self.CHUNK_SIZE

        chunk = self.get_chunk(cx, cy)

        local_x = tx % self.CHUNK_SIZE

        local_y = ty % self.CHUNK_SIZE

        return chunk["tiles"][local_y][local_x]
    
    def isWalkable(self, x, y):
        """ По координатам возвращает, можно ли идти на эту клетку """
        tile_x = int(x // self.TILE_SIZE)
        tile_y = int(y // self.TILE_SIZE)

        tile = self.get_tile(
            tile_x,
            tile_y
        )

        return tile[1]
    

    def get_chunks_around(self, x, y, radius=1):
        """ Получить чанки вокруг координат (x, y) в радиусе radius (в чанках) """
        # Получаем координаты центрального тайла
        tile_x = int(x // self.TILE_SIZE)
        tile_y = int(y // self.TILE_SIZE)
        # Получаем координаты центрального чанка   
        center_cx = tile_x // self.CHUNK_SIZE
        center_cy = tile_y // self.CHUNK_SIZE

        chunks = []

        for cy in range(
            center_cy - radius,
            center_cy + radius + 1
        ):

            for cx in range(
                center_cx - radius,
                center_cx + radius + 1
            ):

                # chunks.append(
                #     self.get_chunk(cx, cy)
                # )
                chunk = self.get_chunk(cx, cy)

                if chunk:
                    chunks.append(chunk)

        return chunks
