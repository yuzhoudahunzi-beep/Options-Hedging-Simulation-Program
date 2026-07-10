"""
FastAPI 服务器 — WebSocket + REST API + 静态文件

端点:
  GET  /           → 静态 HTML 页面
  WS   /ws        → WebSocket 实时通信
  POST /api/action → REST API（备用）
  GET  /api/state  → 获取当前状态
"""

import sys
import os

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from server.game_engine import GameEngine

# ============ FastAPI App ============
app = FastAPI(title="期权对冲训练系统", version="2.0")

# CORS（开发用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局游戏引擎实例
game = GameEngine()


# ============ WebSocket ============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点"""
    await game.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            result = await game.handle_action(data)
            await websocket.send_json({"type": "action_result", **result})
    except WebSocketDisconnect:
        game.disconnect(websocket)
    except Exception as e:
        print(f"[WS] 连接错误: {e}")
        game.disconnect(websocket)


# ============ REST API ============

class ActionRequest(BaseModel):
    type: str
    qty: Optional[int] = None
    option_id: Optional[int] = None
    underlying: Optional[str] = "50ETF"
    difficulty: Optional[str] = "medium"
    mode: Optional[str] = "simulate"
    model: Optional[str] = "heston"
    total_days: Optional[int] = 60
    initial_cash: Optional[float] = 100000
    speed: Optional[float] = 1.0
    seed: Optional[int] = None


@app.post("/api/action")
async def api_action(req: ActionRequest):
    """REST API 操作端点"""
    result = await game.handle_action(req.dict())
    return result


@app.get("/api/state")
async def api_state():
    """获取当前游戏状态"""
    state = {
        "game_state": game.state.value,
        "game": {
            "day": game.game_day,
            "total_days": game.total_days,
            "difficulty": game.difficulty,
        },
    }
    if game.market:
        state["market"] = game.market.get_state()
    if game.portfolio and game.market:
        state["portfolio"] = game.portfolio.get_full_state(
            game.market.S, game.market.iv, game.tick_count
        )
    return state


# ============ 静态文件 ============

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>期权对冲训练系统</h1><p>静态文件未找到</p>")


@app.get("/static/{path}")
async def static_file(path: str):
    """静态文件服务"""
    file_path = os.path.join(STATIC_DIR, path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "Not found"}


# 挂载静态文件目录
if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="static")


# ============ 入口 ============

def main():
    """启动服务器"""
    import uvicorn
    print("=" * 50)
    print("  📊 期权对冲交易员训练系统 v2.0")
    print("  🌐 打开浏览器访问: http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
