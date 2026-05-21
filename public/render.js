// render.js
// import { TILE_SIZE, getTile, getTileColor } from './terrain.js';

export function createRenderer(game) {


    const ctx = game.ctx;
    const canvas = game.canvas;

    function drawFallbackPlayer(isMe) {

        ctx.fillStyle = isMe ? '#0f0' : '#a0f';

        ctx.beginPath();
        ctx.arc(0, 0, 12, 0, Math.PI * 2);
        ctx.fill();
    }

    function drawSpritePlayer(p) {

        const moving =
            Math.abs(p.vx || 0) > 0.05 ||
            Math.abs(p.vy || 0) > 0.05 ||
            p.isMoving || false;

        let frame;

        if (moving) {

            const walkFrame =
                Math.floor(game.animationTime / 10)
                % game.SPRITES.player.walk.length;

            frame = game.SPRITES.player.walk[walkFrame];

        } else {

            frame = game.SPRITES.player.idle[0];
        }

        const [sx, sy, sw, sh] = frame;

        ctx.drawImage(
            game.playerSheet,
            sx,
            sy,
            sw,
            sh,
            -16,
            -16,
            32,
            32
        );
    }

    function getHealthColor(hpPercent) {

        hpPercent = Math.max(0, Math.min(1, hpPercent));

        const hue = hpPercent * 120;

        return `hsl(${hue}, 100%, 45%)`;
    }

    function drawHealthBar(x, y, hp, maxWidth = 32) {

        const hpPercent = Math.max(0, Math.min(1, hp / 100));

        const barWidth = maxWidth * hpPercent;

        const barHeight = 4;

        const barY = y - 42;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(
            x - maxWidth / 2 - 1,
            barY - 1,
            maxWidth + 2,
            barHeight + 2
        );

        ctx.fillStyle = getHealthColor(hpPercent);

        ctx.fillRect(
            x - maxWidth / 2,
            barY,
            barWidth,
            barHeight
        );

        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 1;

        ctx.strokeRect(
            x - maxWidth / 2,
            barY,
            maxWidth,
            barHeight
        );

        if (hp < 25 && Math.floor(Date.now() / 150) % 2 === 0) {

            ctx.fillStyle = 'rgba(255,50,50,0.3)';

            ctx.fillRect(
                x - maxWidth / 2,
                barY,
                maxWidth,
                barHeight
            );
        }
    }

    function drawName(p) {

        ctx.fillStyle = '#fff';

        ctx.font = '12px monospace';

        ctx.textAlign = 'center';

        ctx.fillText(
            p.name || 'Player',
            p.x - game.camera.x,
            p.y - game.camera.y - 24
        );
    }

    function renderBackground() {
        // Черный фон как фон
        // ctx.fillStyle = '#000';
        // ctx.fillRect(
        //     0,
        //     0,
        //     canvas.width,
        //     canvas.height
        // );

        renderTerrain() 
    }

    function renderGrid() {
        ctx.strokeStyle = '#080808';
        ctx.lineWidth = 1;
        const GRID = 50;
        const offsetX =
            -game.camera.x % GRID;
        const offsetY =
            -game.camera.y % GRID;

        for(let x = offsetX; x < canvas.width; x += GRID) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }

        for(let y = offsetY; y < canvas.height; y += GRID) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
    }

    function renderEnemies() {

        game.state.enemies?.forEach(e => {
            ctx.save();
            ctx.translate(e.x - game.camera.x, e.y - game.camera.y);
            ctx.rotate(e.angle - Math.PI / 2);
            if(!game.spritesLoaded) {
                drawFallbackPlayer(isMe);
            } else {
                drawSpritePlayer(e);
            }
            ctx.restore();
            drawName(e);
            drawHealthBar(
                e.x - game.camera.x,
                e.y - game.camera.y,
                e.hp,
                32
            );
        });
    }

    function renderPlayers() {

        game.state.players?.forEach(p => {

            const isMe = p.id === game.myId;

            ctx.save();
            ctx.translate(p.x - game.camera.x, p.y - game.camera.y);
            ctx.rotate(p.angle - Math.PI / 2);
            if(!game.spritesLoaded) {
                drawFallbackPlayer(isMe);
            } else {
                drawSpritePlayer(p);
            }
            ctx.restore();
            drawName(p);
            drawHealthBar(
                p.x - game.camera.x,
                p.y - game.camera.y,
                p.hp,
                32
            );
        });
    }

    function renderBullets() {

        game.state.bullets?.forEach(b => {

            const bullet = game.SPRITES.bullet;

            ctx.save();

            ctx.translate(b.x - game.camera.x, b.y - game.camera.y);

            ctx.rotate(Math.atan2(b.vy, b.vx));

            ctx.drawImage(
                game.playerSheet,

                bullet[0],
                bullet[1],
                bullet[2],
                bullet[3],

                -8,
                -3,
                16,
                6
            );

            ctx.restore();
        });
    }

    function renderEffects() {

        game.state.hits?.forEach(h => {

            const frame =
                game.SPRITES.hit[
                    Math.floor(Date.now() / 40) % 3
                ];

            ctx.save();

            ctx.translate(h.x - game.camera.x, h.y - game.camera.y);

            ctx.drawImage(
                game.playerSheet,

                frame[0],
                frame[1],
                frame[2],
                frame[3],

                -12,
                -12,
                24,
                24
            );

            ctx.restore();
        });

        game.localEffects = game.localEffects.filter(effect => {

            if(effect.type === 'muzzle') {

                const frame =
                    game.SPRITES.muzzle[
                        Math.floor(Date.now() / 40) % 3
                    ];

                ctx.save();

                ctx.translate(effect.x - game.camera.x, effect.y - game.camera.y);

                ctx.rotate(effect.angle);

                ctx.globalAlpha =
                    effect.life /
                    game.EFFECTS.muzzleFlash.life;

                ctx.drawImage(
                    game.playerSheet,

                    frame[0],
                    frame[1],
                    frame[2],
                    frame[3],

                    -12,
                    -12,
                    24,
                    24
                );

                ctx.restore();

                ctx.globalAlpha = 1;

                effect.life -= 1 / 60;

                return effect.life > 0;
            }

            return false;
        });
    }

    function renderUI() {

        if(game.spectatorMode) {

            document.getElementById('ui').style.display = 'none';

        } else {

            const player = game.me();

            document.getElementById('hp').textContent =
                player?.hp || 100;

            document.getElementById('score').textContent =
                player?.score || 0;
        }
    }

    function renderTerrain() {
        const chunks =
            game.state.chunks || [];
        for(const chunk of chunks) {
            for(let y = 0; y < chunk.tiles.length; y++) {
                for(let x = 0; x < chunk.tiles[y].length; x++) {
                    const tile =
                        chunk.tiles[y][x];
                    const ground =
                        tile[0];
                    let color = "#163616";
                    if(ground === 1)
                        color = "#333";
                    if(ground === 2)
                        color = "#1f4a1f";
                    if(ground === 3)
                        color = "#103050";
                    ctx.fillStyle = color;
                    const CHUNK_SIZE =
                        game.state.world?.chunk_size || 32;
                    const TILE_SIZE =
                        game.state.world?.tile_size || 32;
                    const worldX =
                        (
                            chunk.cx * CHUNK_SIZE + x
                        ) * TILE_SIZE;
                    const worldY =
                        (
                            chunk.cy * CHUNK_SIZE + y
                        ) * TILE_SIZE;
                    ctx.fillRect(
                        worldX - game.camera.x,
                        worldY - game.camera.y,
                        TILE_SIZE,
                        TILE_SIZE
                    );
                }
            }
        }
    }

    function updateCamera() {
        const player = game.me();
        if(!player) return;
        // game.camera.x =
        //     player.x - canvas.width / 2;
        // game.camera.y =
        //     player.y - canvas.height / 2;

        // Плавное движение камеры
        const targetX =
            player.x - canvas.width / 2;

        const targetY =
            player.y - canvas.height / 2;

        game.camera.x +=
            (targetX - game.camera.x) * 0.01;

        game.camera.y +=
            (targetY - game.camera.y) * 0.01;
    }

    function render() {

        game.animationTime++;
        updateCamera();
        renderBackground();
        // renderGrid();
        renderEnemies();
        renderPlayers();
        renderBullets();
        renderEffects();
        renderUI();

        requestAnimationFrame(render);
    }

    return {
        render
    };
}