#!/usr/bin/env bash
# ============================================================================
#  一键启动脚本 — Redis + Celery Workers + Flower + Backend
#  用法: bash /root/web/start_all.sh
#  查看: screen -r web-platform
#        Ctrl+A N    切换下一窗口
#        Ctrl+A "    窗口列表
#        Ctrl+A D    分离 (后台运行)
# ============================================================================
set -euo pipefail

SESSION="web-platform"
VENV_BIN="/root/web/envs/bin"
BACKEND_DIR="/root/web/backend"
WORKER_DIR="/root/web/fragtree_workers"

# ── 清理旧会话 ──────────────────────────────────────────────────────────
if screen -ls 2>/dev/null | grep -q "\.${SESSION}"; then
    echo "[!] 发现已有会话 $SESSION，正在关闭..."
    screen -S "$SESSION" -X quit 2>/dev/null || true
    sleep 1
fi

# ── 启动 Redis ──────────────────────────────────────────────────────────
echo "[1/4] 启动 Redis ..."
if command -v redis-server &>/dev/null; then
    redis-server --daemonize yes 2>/dev/null || true
    sleep 1
    if redis-cli ping &>/dev/null; then
        echo "      Redis OK (PID $(pgrep -x redis-server | head -1))"
    else
        echo "      Redis 启动失败，尝试 service 方式..."
        service redis-server start 2>/dev/null || echo "      Redis 启动失败，请手动启动"
    fi
else
    echo "      redis-server 未找到，跳过"
fi

# ── 创建 screen 会话 ────────────────────────────────────────────────────
echo "[2/4] 创建 screen 会话 ..."
screen -dmS "$SESSION"
sleep 0.3

# ── 窗口 0: Redis 状态 ──────────────────────────────────────────────────
screen -S "$SESSION" -p 0 -X title "redis"
screen -S "$SESSION" -p 0 -X stuff "echo '=== Redis 状态 ==='; redis-cli ping 2>/dev/null || echo 'Redis 未运行'; echo; echo '此窗口空闲，Ctrl+A N 切换'"$'\n'

# ── 窗口 1: Celery Workers ─────────────────────────────────────────────
echo "[3/4] 启动 Celery Workers (窗口 1) ..."
screen -S "$SESSION" -X screen -t "celery-workers"
sleep 0.3
screen -S "$SESSION" -p 1 -X stuff \
  "export PATH=\"${VENV_BIN}:\$PATH\"; cd ${WORKER_DIR}; echo '=== Celery Workers (Fastx2 + Slowx2 + Dynamicx2) ==='; echo; bash start_workers.sh"$'\n'

# ── 窗口 2: Flower ─────────────────────────────────────────────────────
echo "      启动 Flower 监控 (窗口 2) ..."
screen -S "$SESSION" -X screen -t "flower:5555"
sleep 0.3
screen -S "$SESSION" -p 2 -X stuff \
  "export PATH=\"${VENV_BIN}:\$PATH\"; cd ${WORKER_DIR}; echo '=== Flower 监控 ==='; echo 'http://127.0.0.1:5555'; echo; bash start_flower.sh"$'\n'

# ── 窗口 3: Backend ────────────────────────────────────────────────────
echo "[4/4] 启动 Backend (窗口 3) ..."
screen -S "$SESSION" -X screen -t "backend:6008"
sleep 0.3
screen -S "$SESSION" -p 3 -X stuff \
  "export PATH=\"${VENV_BIN}:\$PATH\"; cd ${BACKEND_DIR}; echo '=== FastAPI 后端 ==='; echo 'http://0.0.0.0:6008 /docs'; echo; python main.py"$'\n'

# ── 完成 ────────────────────────────────────────────────────────────────
echo
echo "═════════════════════════════════════════════════════════"
echo "  screen 会话: $SESSION"
echo "═════════════════════════════════════════════════════════"
echo "  [0] Redis 状态"
echo "  [1] Celery Workers"
echo "  [2] Flower  →  http://127.0.0.1:5555"
echo "  [3] Backend  →  http://127.0.0.1:6008"
echo "═════════════════════════════════════════════════════════"
echo
echo "  screen -r $SESSION    附加"
echo "  Ctrl+A D              分离"
echo "  screen -S $SESSION -X quit  关闭全部"
echo
screen -ls 2>/dev/null | grep "${SESSION}" || true
