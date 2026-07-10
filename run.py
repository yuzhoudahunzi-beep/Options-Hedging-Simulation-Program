"""
期权对冲训练系统 — 一键启动脚本

用法: python run.py
"""
import os
import sys
import webbrowser
import threading
import time

# 确保项目目录在 path 中
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(PROJECT_DIR, "option_hedge_simulator")
sys.path.insert(0, SIM_DIR)
os.chdir(SIM_DIR)


def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)
    url = "http://127.0.0.1:8000"
    print(f"\n  Opening browser: {url}\n")
    webbrowser.open(url)


def main():
    print("=" * 50)
    print("  Option Hedge Trainer v2.0")
    print("=" * 50)
    print()

    # 检查依赖
    try:
        import fastapi
        import uvicorn
        import numpy
        import scipy
    except ImportError as e:
        print(f"  Installing missing dependencies...")
        os.system(f"{sys.executable} -m pip install fastapi 'uvicorn[standard]' websockets numpy scipy pandas -q")
        print()

    # 启动浏览器（后台线程）
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    # 启动服务器
    print(f"  Server: http://127.0.0.1:8000")
    print(f"  Press Ctrl+C to stop")
    print()

    uvicorn.run(
        "server.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
