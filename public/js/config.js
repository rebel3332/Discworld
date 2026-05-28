export async function loadConfig(path) {

    const response = await fetch(path);

    if(!response.ok) {
        throw new Error(
            `Failed to load config: ${path}`
        );
    }

    return await response.json();
}

export async function loadAllConfigs() {

    const [
        sprites,
        effects,
        entities,
        world,
        tiles,
        sensors
    ] = await Promise.all([

        loadConfig('/static/config/sprites.json'),
        loadConfig('/static/config/effects.json'),
        loadConfig('/static/config/entities.json'),
        loadConfig('/static/config/world.json'),
        loadConfig('/static/config/tiles.json'),
        loadConfig('/static/config/sensors.json')
    ]);

    return {
        sprites,
        effects,
        entities,
        world,
        tiles,
        sensors
    };
}