#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def convert(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError('JSON must contain a "models" array')

    for i, model in enumerate(models):
        slug = model.get("slug", f"models[{i}]")

        # base_instructions
        if "base_instructions" not in model:
            model_messages = model.get("model_messages")
            if not isinstance(model_messages, dict):
                raise ValueError(f"{slug}: missing model_messages")

            instructions = model_messages.get("instructions_template")
            if not isinstance(instructions, str):
                raise ValueError(
                    f"{slug}: missing model_messages.instructions_template"
                )

            model["base_instructions"] = instructions

        # supports_reasoning_summaries
        if "supports_reasoning_summaries" not in model:
            # The old catalog already has supports_reasoning_summary_parameter.
            model["supports_reasoning_summaries"] = model.get(
                "supports_reasoning_summary_parameter",
                False,
            )

    # Validate required fields
    for i, model in enumerate(models):
        slug = model.get("slug", f"models[{i}]")

        for field in (
            "base_instructions",
            "supports_reasoning_summaries",
        ):
            if field not in model:
                raise ValueError(f"{slug}: missing required field {field}")

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Converted {len(models)} models")
    print(f"Output: {output_file}")


def main():
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
        p = Path(input_path)
        output_path = str(
            p.with_name(p.stem + ".converted" + p.suffix)
        )

    convert(input_path, output_path)


if __name__ == "__main__":
    main()
