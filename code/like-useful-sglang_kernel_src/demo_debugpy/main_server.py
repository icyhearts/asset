#!/usr/bin/env python3
"""
主进程: 启动多个 Worker 子进程, 分发任务, 收集结果。

用法:
    # 不带调试 (直接跑)
    python main_server.py

    # Worker-0 开启 debugpy, 监听 5678 端口
    python main_server.py --debug-workers 0 --debug-port-base 5678

    # Worker-0 和 Worker-1 都开启 debugpy
    python main_server.py --debug-workers 0,1 --debug-port-base 5678
    # Worker-0 监听 5678, Worker-1 监听 5679
"""

import argparse
import multiprocessing as mp
import os
import time

from worker import worker_main


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-process debugpy demo")
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--debug-workers",
        type=str,
        default="",
        help="Comma-separated worker IDs to debug, e.g., '0,1' (empty = no debug)",
    )
    parser.add_argument(
        "--debug-port-base",
        type=int,
        default=5678,
        help="Base port for debugpy. Worker-i uses port base+i",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析要调试的 worker ID
    debug_worker_ids = set()
    if args.debug_workers:
        debug_worker_ids = set(
            int(x.strip()) for x in args.debug_workers.split(",")
        )

    print(f"[Main PID={os.getpid()}] starting {args.num_workers} workers", flush=True)
    if debug_worker_ids:
        print(
            f"[Main] Debug enabled for workers: {sorted(debug_worker_ids)}",
            flush=True,
        )

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # ---- 启动子进程 ---- #
    workers = []
    for i in range(args.num_workers):
        debug_port = args.debug_port_base + i if i in debug_worker_ids else 0
        p = mp.Process(
            target=worker_main,
            args=(task_queue, result_queue, i, debug_port),
        )
        p.start()
        workers.append(p)
        if debug_port > 0:
            print(
                f"[Main] Worker-{i} started, PID={p.pid}, debugpy port={debug_port}",
                flush=True,
            )
        else:
            print(
                f"[Main] Worker-{i} started, PID={p.pid}, no debug",
                flush=True,
            )

    if debug_worker_ids:
        lines = []
        for wid in sorted(debug_worker_ids):
            port = args.debug_port_base + wid
            lines.append(f"  Worker-{wid} waiting on port {port}")
        print(
            f"\n{'='*60}\n"
            + "\n".join(lines)
            + f"\n  Open nvim, run :lua require'dap'.continue()\n"
            f"  Select the worker you want to attach\n"
            f"{'='*60}\n",
            flush=True,
        )

    # ---- 分发任务 ---- #
    num_tasks = 6
    for task_id in range(num_tasks):
        value = (task_id + 1) * 10
        task_queue.put((task_id, value))
        print(f"[Main] dispatched task {task_id}, value={value}", flush=True)

    # ---- 收集结果 ---- #
    for _ in range(num_tasks):
        task_id, result = result_queue.get()
        print(f"[Main] got result: task {task_id} => {result}", flush=True)

    # ---- 关闭 ---- #
    for _ in range(args.num_workers):
        task_queue.put(None)  # 毒丸

    for p in workers:
        p.join()

    print("[Main] all workers finished", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")  # 和 CUDA 程序一致, 用 spawn
    main()
