"""Reading verl's dataset format: one parquet row per task.

verl rows carry `data_source`, `prompt` (chat messages), `ability`, `reward_model`
and `extra_info`. Agentic datasets put the environment in `extra_info.instance_json`,
a JSON string; that object is what each adapter reads.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

MIMO_REPO = "XiaomiMiMo/MiMo-V2.6-RL-oss"
# the README's image repository; a row's docker_image is the tag within it
MIMO_IMAGES = "xiaomimimo/mimo-v2.6-rl-oss"


def read_rows(source: str) -> list[dict[str, Any]]:
    """The `instance_json` object of every row in a verl parquet.

    `source` is a local path, or `<hf dataset repo>/<file>` for a file on the Hub,
    e.g. `XiaomiMiMo/MiMo-V2.6-RL-oss/code.parquet`.
    """
    path = Path(source).expanduser()
    if not path.exists():
        from huggingface_hub import hf_hub_download

        owner, name, filename = source.split("/", 2)
        path = Path(hf_hub_download(f"{owner}/{name}", filename, repo_type="dataset"))
    frame = pd.read_parquet(path)
    return [json.loads(extra["instance_json"]) for extra in frame["extra_info"]]


def mimo_image(docker_image: str) -> str:
    """The pullable image for a MiMo row: `format-code-task-001457:latest` names a tag."""
    if "/" in docker_image:
        return docker_image
    return f"{MIMO_IMAGES}:{docker_image.split(':')[0]}"
