import os
import json
import pytest
from utils import *

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

server = ServerPreset.stories15m_moe()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]
    server.slot_save_path = "./tmp"
    server.n_slots = 2


def test_slot_save_lora_sidecar_written():
    """Saving a slot with an active LoRA should write a .lora.json sidecar
    alongside the state file, recording the adapter path and scale."""
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
        "id_slot": 0,
        "lora": [{"id": 0, "scale": 0.5}],
        "cache_prompt": True,
    })
    assert res.status_code == 200

    filename = "slot_lora_sidecar_test.bin"
    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": filename,
    })
    assert res.status_code == 200

    sidecar_path = os.path.join("./tmp", filename + ".lora.json")
    assert os.path.exists(sidecar_path), f"LoRA sidecar not found: {sidecar_path}"

    with open(sidecar_path) as f:
        sidecar = json.load(f)

    assert sidecar.get("lora_sidecar_version") == 1
    assert sidecar.get("alora_invocation_start") == -1
    assert len(sidecar.get("adapters", [])) == 1
    assert sidecar["adapters"][0]["scale"] == pytest.approx(0.5)
    assert sidecar["adapters"][0]["path"] != ""


def test_slot_save_restore_lora_cache_preserved():
    """Restoring a slot saved with LoRA scale=0.5 and then requesting with the
    same scale should reuse the KV cache (prompt_n < full prompt length).

    Without the .lora.json sidecar the restored slot would carry the default
    scale (1.0 from --lora), causing are_lora_equal() to fail and the cache
    to be fully cleared before inference.
    """
    global server
    server.start()

    prompt = "Look in thy glass"

    # First request on slot 1: all prompt tokens are processed from scratch
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 1,
        "lora": [{"id": 0, "scale": 0.5}],
        "cache_prompt": True,
    })
    assert res.status_code == 200
    first_prompt_n = res.body["timings"]["prompt_n"]
    assert first_prompt_n > 0

    # Save slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_lora_cache_test.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] > 0

    # Restore into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_lora_cache_test.bin",
    })
    assert res.status_code == 200

    # Same prompt, same LoRA scale — sidecar restores scale=0.5 so
    # are_lora_equal() returns true and the KV cache is reused
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "lora": [{"id": 0, "scale": 0.5}],
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] < first_prompt_n


def test_slot_save_restore_lora_cache_invalidated_on_scale_change():
    """Restoring a slot saved with LoRA scale=0.5 and then requesting with a
    different scale should fully clear the cache (prompt_n == full prompt
    length) regardless of the sidecar."""
    global server
    server.start()

    prompt = "Look in thy glass"

    # First request on slot 1
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 1,
        "lora": [{"id": 0, "scale": 0.5}],
        "cache_prompt": True,
    })
    assert res.status_code == 200
    first_prompt_n = res.body["timings"]["prompt_n"]
    assert first_prompt_n > 0

    # Save slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_lora_scale_change_test.bin",
    })
    assert res.status_code == 200

    # Restore into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_lora_scale_change_test.bin",
    })
    assert res.status_code == 200

    # Same prompt but different LoRA scale — are_lora_equal() fails and
    # lora_should_clear_cache() triggers a full cache wipe
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "lora": [{"id": 0, "scale": 1.0}],
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == first_prompt_n
