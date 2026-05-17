// render.js

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
            p.x,
            p.y - 24
        );
    }

    function renderBackground() {

        ctx.fillStyle = '#000';

        ctx.fillRect(
            0,
            0,
            canvas.width,
            canvas.height
        );
    }

    function renderGrid() {

        ctx.strokeStyle = '#080808';

        ctx.lineWidth = 1;

        for(let x = 0; x < canvas.width; x += 50) {

            ctx.beginPath();

            ctx.moveTo(x, 0);

            ctx.lineTo(x, canvas.height);

            ctx.stroke();
        }

        for(let y = 0; y < canvas.height; y += 50) {

            ctx.beginPath();

            ctx.moveTo(0, y);

            ctx.lineTo(canvas.width, y);

            ctx.stroke();
        }
    }

    function renderEnemies() {

        game.state.enemies?.forEach(e => {
            ctx.save();
            ctx.translate(e.x, e.y);
            ctx.rotate(e.angle - Math.PI / 2);
            if(!game.spritesLoaded) {
                drawFallbackPlayer(isMe);
            } else {
                drawSpritePlayer(e);
            }
            ctx.restore();
            drawName(e);
            drawHealthBar(
                e.x,
                e.y,
                e.hp,
                32
            );
        });
    }

    function renderPlayers() {

        game.state.players?.forEach(p => {

            const isMe = p.id === game.myId;

            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.angle - Math.PI / 2);
            if(!game.spritesLoaded) {
                drawFallbackPlayer(isMe);
            } else {
                drawSpritePlayer(p);
            }
            ctx.restore();
            drawName(p);
            drawHealthBar(
                p.x,
                p.y,
                p.hp,
                32
            );
        });
    }

    function renderBullets() {

        game.state.bullets?.forEach(b => {

            const bullet = game.SPRITES.bullet;

            ctx.save();

            ctx.translate(b.x, b.y);

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

            ctx.translate(h.x, h.y);

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

                ctx.translate(effect.x, effect.y);

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

    function render() {

        game.animationTime++;

        renderBackground();
        renderGrid();
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