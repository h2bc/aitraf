"""Helpers for Hugging Face model/cache naming."""

from __future__ import annotations


def hf_model_cache_dir_name(repo_id: str) -> str:
    """Return the local Hugging Face cache directory name for a model repo id."""

    return f"models--{repo_id.replace('/', '--')}"


__all__ = ["hf_model_cache_dir_name"]
