#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def convert(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")

    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError('JSON must contain a "models" array')

    converted = 0
    skipped = 0

    for i, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{i}] is not an object")

        # Already converted.
        if "base_instructions" in model:
            skipped += 1
            continue

        model_messages = model.get("model_messages")
        if not isinstance(model_messages, dict):
            raise ValueError(
                f'models[{i}] ({model.get("slug", "<unknown>")}) '
                'has no valid "model_messages" object'
            )

        instructions = model_messages.get("instructions_template")
        if not isinstance(instructions, str):
            raise ValueError(
                f'models[{i}] ({model.get("slug", "<unknown>")}) '
                'has no valid "model_messages.instructions_template"'
            )

        # Put base_instructions at the model level.
        model["base_instructions"] = instructions
        converted += 1

    # Validate before writing.
    for i, model in enumerate(models):
        if "base_instructions" not in model:
            raise ValueError(
                f'models[{i}] ({model.get("slug", "<unknown>")}) '
                'still has no "base_instructions"'
            )

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Input : {input_file}")
    print(f"Output: {output_file}")
    print(f"Models: {len(models)}")
    print(f"Added : {converted}")
    print(f"Skip  : {skipped}")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: {sys.argv[0]} INPUT.json [OUTPUT.json]",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        input_file = Path(input_path)
        output_path = str(
            input_file.with_name(input_file.stem + ".converted" + input_file.suffix)
        )

    convert(input_path, output_path)


if __name__ == "__main__":
    main()
