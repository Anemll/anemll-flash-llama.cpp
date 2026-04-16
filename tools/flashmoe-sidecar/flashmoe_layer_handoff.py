#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import statistics
import struct
import time
from pathlib import Path
from typing import Iterable

MAGIC = b"FMOEHND1"
VERSION = 1
FLAG_ECHO_PAYLOAD = 1 << 0
HEADER = struct.Struct("!8sIIQI")
DEFAULT_PORT = 9501


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the boundary handoff for a planned multi-machine Flash-MoE layer split. "
            "Use 'serve' on the worker side and 'bench' on the driver side."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start a worker-side handoff server")
    serve.add_argument("--bind", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"listen port (default: {DEFAULT_PORT})")
    serve.add_argument("--echo-payload", action="store_true", help="echo payloads back to the client")
    serve.set_defaults(func=cmd_serve)

    bench = sub.add_parser("bench", help="run a driver-side transport benchmark against a worker")
    bench.add_argument("--host", default="127.0.0.1", help="worker host (default: 127.0.0.1)")
    bench.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"worker port (default: {DEFAULT_PORT})")
    bench.add_argument("--plan", type=Path, help="optional plan JSON from flashmoe_layer_split.py")
    bench.add_argument("--payload-bytes", type=int, help="explicit payload size override")
    bench.add_argument("--echo-payload", action="store_true", help="request echoed payloads from the worker")
    bench.add_argument("--warmup", type=int, default=20, help="warmup iterations (default: 20)")
    bench.add_argument("--repeats", type=int, default=200, help="timed iterations (default: 200)")
    bench.add_argument("--seed", type=int, default=123, help="random seed for deterministic payloads")
    bench.set_defaults(func=cmd_bench)

    return parser.parse_args()


def load_plan_payload_bytes(plan_path: Path) -> tuple[int, dict]:
    plan = json.loads(plan_path.expanduser().resolve().read_text(encoding="utf-8"))
    boundary = plan.get("boundary_activation")
    if not isinstance(boundary, dict):
        raise SystemExit(f"plan '{plan_path}' is missing boundary_activation")
    payload_bytes = int(boundary.get("bytes_per_ubatch", 0))
    if payload_bytes <= 0:
        raise SystemExit(f"plan '{plan_path}' has invalid boundary_activation.bytes_per_ubatch={payload_bytes}")
    return payload_bytes, plan


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(min(1 << 20, remaining))
        if not chunk:
            raise ConnectionError("peer closed connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_percentiles(values: list[float], percentiles: Iterable[int]) -> dict[int, float]:
    if not values:
        return {p: 0.0 for p in percentiles}
    ordered = sorted(values)
    out: dict[int, float] = {}
    for p in percentiles:
        if len(ordered) == 1:
            out[p] = ordered[0]
            continue
        rank = max(0.0, min(1.0, p / 100.0)) * (len(ordered) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        alpha = rank - lo
        out[p] = ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha
    return out


def cmd_serve(args: argparse.Namespace) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        print(f"handoff server listening on {args.bind}:{args.port} echo_payload={'on' if args.echo_payload else 'off'}")
        while True:
            conn, addr = server.accept()
            with conn:
                print(f"client connected from {addr[0]}:{addr[1]}")
                while True:
                    try:
                        header = recv_exact(conn, HEADER.size)
                    except ConnectionError:
                        break

                    magic, version, msg_id, payload_bytes, flags = HEADER.unpack(header)
                    if magic != MAGIC:
                        raise SystemExit(f"bad magic {magic!r}")
                    if version != VERSION:
                        raise SystemExit(f"unsupported version {version}")

                    payload = recv_exact(conn, payload_bytes)
                    echo = args.echo_payload or bool(flags & FLAG_ECHO_PAYLOAD)
                    reply_flags = FLAG_ECHO_PAYLOAD if echo else 0
                    conn.sendall(HEADER.pack(MAGIC, VERSION, msg_id, len(payload) if echo else 0, reply_flags))
                    if echo:
                        conn.sendall(payload)
                print("client disconnected")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    plan = None
    if args.plan is not None:
        payload_bytes, plan = load_plan_payload_bytes(args.plan)
    elif args.payload_bytes is not None:
        payload_bytes = int(args.payload_bytes)
    else:
        raise SystemExit("pass either --plan or --payload-bytes")

    rng = random.Random(args.seed)
    payload = bytearray(payload_bytes)
    for i in range(payload_bytes):
        payload[i] = rng.randrange(0, 256)

    flags = FLAG_ECHO_PAYLOAD if args.echo_payload else 0
    timings_ms: list[float] = []
    total_wire_bytes = 0

    with socket.create_connection((args.host, args.port)) as sock:
        for index in range(args.warmup + args.repeats):
            msg_id = index + 1
            t0 = time.perf_counter_ns()
            sock.sendall(HEADER.pack(MAGIC, VERSION, msg_id, payload_bytes, flags))
            sock.sendall(payload)

            reply = recv_exact(sock, HEADER.size)
            magic, version, reply_id, reply_bytes, reply_flags = HEADER.unpack(reply)
            if magic != MAGIC or version != VERSION or reply_id != msg_id:
                raise SystemExit("invalid reply header")
            if reply_bytes > 0:
                _ = recv_exact(sock, reply_bytes)

            t1 = time.perf_counter_ns()
            if index >= args.warmup:
                timings_ms.append((t1 - t0) / 1e6)
                total_wire_bytes += HEADER.size + payload_bytes + HEADER.size + reply_bytes

    p = iter_percentiles(timings_ms, [50, 90, 99])
    avg_ms = statistics.fmean(timings_ms) if timings_ms else 0.0
    total_seconds = sum(timings_ms) / 1000.0
    gib_per_s = (total_wire_bytes / float(1024 ** 3)) / total_seconds if total_seconds > 0 else 0.0

    print(f"payload_bytes={payload_bytes}")
    if plan is not None:
        model = plan.get("model", {})
        boundary = plan.get("boundary_activation", {})
        print(
            "plan="
            f"arch:{model.get('arch')} "
            f"embd:{model.get('embedding_length')} "
            f"ubatch:{boundary.get('ubatch')} "
            f"mib_per_ubatch:{boundary.get('mib_per_ubatch')}"
        )
    print(f"echo_payload={'on' if args.echo_payload else 'off'}")
    print(f"repeats={args.repeats} warmup={args.warmup}")
    print(f"p50_ms={p[50]:.4f}")
    print(f"p90_ms={p[90]:.4f}")
    print(f"p99_ms={p[99]:.4f}")
    print(f"avg_ms={avg_ms:.4f}")
    print(f"wire_gib_per_s={gib_per_s:.4f}")
    return 0


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
