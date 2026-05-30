import json


def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SENSORS = load_json(
    "public/config/sensors.json"
)

WORLDS = load_json(
    "public/config/world.json"
)