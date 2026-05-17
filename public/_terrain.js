// terrain.js

export const TILE_SIZE = 32;
export const WORLD_WIDTH = 4000;
export const WORLD_HEIGHT = 4000;

export function getTile(tx, ty) {

    const v =
        Math.sin(tx * 0.15)
        + Math.cos(ty * 0.15);

    if(v < -0.5) return "stone";

    if(v < 0.5) return "grass";

    return "toxic";
}

export function getTileColor(tile) {

    switch(tile) {

        case "grass":
            return "#163616";

        case "toxic":
            return "#1f4a1f";

        case "stone":
            return "#222";

        default:
            return "#000";
    }
}