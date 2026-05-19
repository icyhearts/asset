"""
Worker 子进程: 模拟 SGLang 的 TP Worker。
主进程无法用调试器直接断点到这里, 需要用 debugpy attach。
"""

import os
import sys
import time
import debugpy


def init_debugpy_in_worker(port=5678):
    """在子进程中初始化 debugpy, 等待调试器 attach。"""
    debugpy.listen(("0.0.0.0", port))
    print(
        f"[Worker PID={os.getpid()}] debugpy listening on port {port}, "
        f"waiting for debugger to attach...",
        flush=True,
    )
    debugpy.wait_for_client()  # 阻塞, 直到调试器连接
    print(f"[Worker PID={os.getpid()}] debugger attached!", flush=True)


def heavy_compute(x):
    """模拟 forward_decode 之类的计算函数。"""
    result = 0
    for i in range(x):
        result += i * i  # ← 可以在这里打断点
    return result


def worker_main(task_queue, result_queue, worker_id, debug_port=0):
    """
    子进程入口。
    debug_port > 0 时启用 debugpy。
    """
    print(f"[Worker-{worker_id} PID={os.getpid()}] started", flush=True)

    if debug_port > 0:
        init_debugpy_in_worker(port=debug_port)

    while True:
        task = task_queue.get()
        if task is None:  # 毒丸, 退出
            print(f"[Worker-{worker_id}] received shutdown signal", flush=True)
            break

        task_id, value = task
        print(
            f"[Worker-{worker_id}] processing task {task_id}, value={value}",
            flush=True,
        )

        # ---- 核心计算 ---- #
        result = heavy_compute(value)  # ← 断点可以设在这行或 heavy_compute 内部
        # ---- 核心计算 ---- #

        result_queue.put((task_id, result))
        print(
            f"[Worker-{worker_id}] task {task_id} done, result={result}",
            flush=True,
        )
