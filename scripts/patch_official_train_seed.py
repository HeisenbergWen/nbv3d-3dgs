#!/usr/bin/env python3
"""Add a --seed option to the official 3DGS train.py script."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SEED_FUNCTION = '''

def set_training_seed(seed: int) -> None:
    import random as _random
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch official gaussian-splatting/train.py with a reproducible "
            "--seed argument. The patch is idempotent and creates a backup by default."
        )
    )
    parser.add_argument("--train-py", type=Path, required=True)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def insert_once(text: str, needle: str, insertion: str, description: str) -> str:
    if insertion.strip() in text:
        return text
    if needle not in text:
        raise RuntimeError(f"Could not find insertion point for {description}: {needle!r}")
    return text.replace(needle, insertion + needle, 1)


def main() -> None:
    args = parse_args()
    train_py = args.train_py.resolve()
    if not train_py.is_file():
        raise FileNotFoundError(f"train.py does not exist: {train_py}")

    text = train_py.read_text(encoding="utf-8")
    original = text

    if "def set_training_seed(seed: int)" not in text:
        text = insert_once(text, "\ndef training(", SEED_FUNCTION + "\n", "set_training_seed")

    if '"--seed"' not in text and "'--seed'" not in text:
        parser_anchor = 'parser.add_argument("--start_checkpoint", type=str, default = None)\n'
        if parser_anchor not in text:
            parser_anchor = "parser.add_argument('--start_checkpoint', type=str, default = None)\n"
        seed_arg = parser_anchor + '    parser.add_argument("--seed", type=int, default=0)\n'
        text = text.replace(parser_anchor, seed_arg, 1)

    if "set_training_seed(args.seed)" not in text:
        safe_state_anchor = "safe_state(args.quiet)\n"
        seed_call = safe_state_anchor + '    set_training_seed(args.seed)\n    print(f"Using random seed: {args.seed}")\n'
        if safe_state_anchor not in text:
            raise RuntimeError("Could not find safe_state(args.quiet) call")
        text = text.replace(safe_state_anchor, seed_call, 1)

    if text == original:
        print(f"No changes needed; {train_py} already has --seed support.")
        return

    if not args.no_backup:
        backup_path = train_py.with_suffix(train_py.suffix + ".before_seed_patch")
        if not backup_path.exists():
            shutil.copy2(train_py, backup_path)
            print(f"Wrote backup: {backup_path}")
    train_py.write_text(text, encoding="utf-8")
    print(f"Patched seed support into: {train_py}")


if __name__ == "__main__":
    main()
