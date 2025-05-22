# OpenAI LLM Benchmark

A quick-and-dirty load-tester for any OpenAI-style LLM endpoint. This tool allows you to benchmark the performance of various LLM models by sending concurrent requests and measuring metrics like latency and tokens per second.

## Features

- Test any OpenAI-compatible API endpoint
- Configure number of requests and concurrency level
- Measure key performance metrics (requests/sec, tokens/sec, latency)
- Support for various models and deployments (vLLM, Ollama, etc.)
- Progress bar visualization (with tqdm)
- Optional response capturing for inspection and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/openai-llm-benchmark.git
cd openai-llm-benchmark

# Install dependencies
uv sync
```

## Usage

```bash
uv run openai-llm-benchmark.py \
       --base-url <API_ENDPOINT> \
       --model <MODEL_NAME> \
       --requests <NUM_REQUESTS> \
       --concurrency <CONCURRENCY_LEVEL>
```

### Example: Testing vLLM

```bash
uv run openai-llm-benchmark.py \
       --base-url http://localhost:8000 \
       --model Qwen/Qwen3-14B \
       --requests 200 --concurrency 12
```

### Example: Testing Ollama

```bash
uv run openai-llm-benchmark.py \
       --base-url http://localhost:11434 \
       --model qwen3:14b-fp16 \
       --requests 200 --concurrency 16
```

### Example: Capturing responses to a file

```bash
uv run openai-llm-benchmark.py \
       --base-url http://localhost:11434 \
       --model qwen3:14b-fp16 \
       --requests 50 --concurrency 8 \
       --capture-responses --output-file results/ollama_responses.json
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--base-url` | API endpoint URL (required) | - |
| `--api-key` | Bearer token for authentication | "" |
| `--model` | Model name to test | "llama3.2" |
| `--prompt` | User prompt to send | "Hello, world!" |
| `--requests` | Total number of requests | 100 |
| `--concurrency` | Number of parallel workers | 10 |
| `--max-tokens` | Maximum tokens per request | 32 |
| `--temperature` | Temperature for sampling (0.0 = deterministic) | 0.2 |
| `--quiet` | Hide progress bar | False |
| `--capture-responses` | Capture LLM responses and write to file | False |
| `--output-file` | File path for captured responses | "responses.json" |

## Output

The benchmark will output:
- Number of successful requests
- Total execution time
- Requests per second
- Tokens per second (if available)
- Average latency
- p50 latency (median)
- p95 latency

When `--capture-responses` is enabled, all LLM responses will be written to the specified output file in JSON format.

## Requirements

- Python 3.12+
- httpx[http2]
- numpy
- tqdm (optional, for progress bar)