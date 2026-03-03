"""
Local conftest for unit tests.

Overrides the module-level do_something fixture from the parent conftest.py
so that tests can run without downloading models from HuggingFace.
All ServerPreset factory methods are monkey-patched to use the local
tiny GGUF model file at TINY_MODEL_PATH.
"""
import os
import pytest
from utils import ServerPreset, ServerProcess

TINY_MODEL_PATH = "/tmp/tiny-llama.gguf"
TINY_LORA_PATH  = "/tmp/tiny-llama-lora.gguf"


def _make_tiny_server() -> ServerProcess:
    server = ServerProcess()
    server.model_file    = TINY_MODEL_PATH
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_alias   = "tiny-llama"
    server.n_ctx         = 512
    server.n_batch       = 64
    server.n_slots       = 1
    server.n_predict     = 8
    server.temperature   = 0.0
    server.seed          = 42
    server.offline       = True
    return server


def _patch_presets():
    """Replace all ServerPreset factory methods with tiny-model variants."""
    for name, method in list(ServerPreset.__dict__.items()):
        if callable(method) and name not in ("load_all", "__init__"):
            setattr(ServerPreset, name, staticmethod(_make_tiny_server))


_patch_presets()


@pytest.fixture(scope="module", autouse=True)
def do_something():
    """Override parent conftest do_something — skip load_all() entirely."""
    yield
