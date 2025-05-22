#!/usr/bin/env python3
"""
benchmark_openai.py

Quick-and-dirty load-tester for any OpenAI-style LLM endpoint.
Example:

# vLLM running on port 8000
python benchmark_openai.py \
       --base-url http://localhost:8000 \
       --model Qwen/Qwen3-14B \
       --requests 200 --concurrency 12

# Ollama (must be 0.1.34+ which exposes /v1/chat/completions)
python benchmark_openai.py \
       --base-url http://localhost:11434 \
       --model qwen3:14b-fp16 \
       --requests 200 --concurrency 16
"""

import argparse
import asyncio
import statistics
import time
from typing import Tuple, List, Optional
import httpx
import numpy as np
try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError: 
    tqdm_asyncio = None

async def _chat_completion(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
) -> Tuple[Optional[float], Optional[int]]:
    """Send one completion request and return (latency, total_tokens)."""
    t0 = time.perf_counter()
    try:
        r = await client.post(url, headers=headers, json=payload, timeout=60)
        latency = time.perf_counter() - t0
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        tokens = usage.get(
            "total_tokens",
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        )
        return latency, tokens
    except Exception:
        return None, None

async def _run_once(args) -> None:
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        **({"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}),
    }
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
    }

    sem = asyncio.Semaphore(args.concurrency)
    latencies: List[float] = []
    tokens: List[int] = []

    async def worker():
        async with sem:
            l, t = await _chat_completion(client, url, headers, payload)
            if l is not None:
                latencies.append(l)
                tokens.append(t)

    async with httpx.AsyncClient(http2=True, timeout=None) as client:
        # Optional one-shot warm-up
        await _chat_completion(client, url, headers, payload)

        tic = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(args.requests)]
        if tqdm_asyncio and not args.quiet:
            await tqdm_asyncio.gather(*tasks)
        else:
            await asyncio.gather(*tasks)
        toc = time.perf_counter()

    _report(latencies, tokens, args.requests, toc - tic)

# Report
def _report(latencies: List[float], tokens: List[int], total_req: int, wall: float):
    ok = len(latencies)
    print(f"\n✔ {ok}/{total_req} requests succeeded in {wall:.2f}s")
    if ok == 0:
        return
    rps = ok / wall
    print(f"Requests/s:     {rps:10.2f}")
    if any(tokens):
        tps = sum(tokens) / wall
        print(f"Tokens/s:       {tps:10.2f}")
    print(f"Avg latency:    {statistics.mean(latencies):10.3f}s")
    print(f"p50 latency:    {np.percentile(latencies, 50):10.3f}s")
    print(f"p95 latency:    {np.percentile(latencies, 95):10.3f}s\n")


# CLI
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concurrent benchmark for OpenAI-style LLMs")
    p.add_argument("--base-url", required=True, help="http(s)://host:port")
    p.add_argument("--api-key", default="", help="Bearer token if your server needs one")
    p.add_argument("--model", default="llama3.2", help="Model name")
    p.add_argument("--prompt", default="Hello, world!", help="User prompt")
    p.add_argument("--requests", type=int, default=100, help="Total number of requests")
    p.add_argument("--concurrency", type=int, default=10, help="Parallel workers")
    p.add_argument("--max-tokens", type=int, default=32, help="max_tokens per request")
    p.add_argument("--quiet", action="store_true", help="Hide progress bar")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    asyncio.run(_run_once(args))

