"""CLI entrypoint: python -m server [--flags].

Every flag falls back to an IE_* environment variable, so the server can be
configured entirely from the environment (see .env.example).
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from engine.config import EngineConfig
from server.app import create_app


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m server",
                                description="OpenAI-compatible inference server")
    p.add_argument("--model", default=_env("IE_MODEL", "gpt2"))
    p.add_argument("--draft-model", default=_env("IE_DRAFT_MODEL", "") or None,
                   help="enable speculative decoding with this draft model")
    p.add_argument("--block-size", type=int, default=int(_env("IE_BLOCK_SIZE", "16")))
    p.add_argument("--num-blocks", type=int, default=int(_env("IE_NUM_BLOCKS", "512")))
    p.add_argument("--max-batch-size", type=int, default=int(_env("IE_MAX_BATCH_SIZE", "8")))
    p.add_argument("--max-model-len", type=int, default=int(_env("IE_MAX_MODEL_LEN", "1024")))
    p.add_argument("--num-speculative-tokens", type=int,
                   default=int(_env("IE_NUM_SPECULATIVE_TOKENS", "4")))
    p.add_argument("--host", default=_env("IE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(_env("IE_PORT", "8000")))
    args = p.parse_args()

    config = EngineConfig(model=args.model, draft_model=args.draft_model,
                          block_size=args.block_size, num_blocks=args.num_blocks,
                          max_batch_size=args.max_batch_size,
                          max_model_len=args.max_model_len,
                          num_speculative_tokens=args.num_speculative_tokens)
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
