import json
from pathlib import Path

CHECKPOINT_PATH = Path("data/eval/index_checkpoint.json")


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {
        "completed_docs": []
    }


def save_checkpoint(state):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(state, indent=2))