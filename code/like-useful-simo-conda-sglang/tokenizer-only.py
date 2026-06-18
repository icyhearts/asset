#!/usr/bin/env python3
import argparse
from typing import Iterable, Optional

from transformers import AutoTokenizer


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in ("1", "true", "t", "yes", "y"):
        return True
    if lowered in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_optional_bool(value: str) -> Optional[bool]:
    if value.lower() in ("none", "null"):
        return None
    return parse_bool(value)


def has_bos_prefix(sequence: str, bos_str: Optional[str | Iterable[str]]) -> bool:
    if bos_str is None:
        return False
    if isinstance(bos_str, str):
        return sequence.startswith(bos_str)
    return any(sequence.startswith(item) for item in bos_str)


def add_special_kwargs(
    add_special_tokens: Optional[bool],
    add_bos: Optional[bool],
) -> dict:
    if add_special_tokens is not None:
        return {"add_special_tokens": add_special_tokens}
    if add_bos is not None:
        return {"add_special_tokens": add_bos}
    return {}


def sglang_tok_encode(tokenizer, text: str, add_bos_token: bool) -> list[int]:
    # Mirrors lm_eval.models.sglang_causallms.SGLangLM.tok_encode.
    add_special_tokens = False
    if not add_special_tokens:
        add_special_tokens = False or add_bos_token
    return tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
    ).input_ids


def vllm_tok_encode(
    tokenizer,
    text: str,
    prefix_token_id: int,
    add_bos_token: Optional[bool],
    add_special_tokens: Optional[bool],
) -> list[int]:
    # Mirrors the relevant path in lm_eval.models.vllm_causallms.VLLM.tok_encode.
    bos_token = tokenizer.decode(prefix_token_id)
    kwargs = add_special_kwargs(add_special_tokens, add_bos_token)
    if has_bos_prefix(text, bos_token):
        kwargs = {**kwargs, "add_special_tokens": False}
    return tokenizer(text, return_attention_mask=False, **kwargs).input_ids


def first_disjoint_rolling_window(
    token_list: list[int],
    prefix_token: int,
    max_seq_len: int,
) -> tuple[list[int], list[int], list[int]]:
    # Mirrors get_rolling_token_windows + make_disjoint_window for the first window.
    if not token_list:
        return [], [], []
    first_seq_len = min(max_seq_len, len(token_list))
    input_tokens = [prefix_token] + token_list[: first_seq_len - 1]
    pred_tokens = token_list[:first_seq_len]
    context = input_tokens[: len(input_tokens) - (len(pred_tokens) - 1)]
    engine_input = context + pred_tokens
    return context, pred_tokens, engine_input


def compact(values: list[int], limit: int) -> str:
    if len(values) <= limit:
        return repr(values)
    head = ", ".join(str(x) for x in values[:limit])
    return f"[{head}, ...] len={len(values)}"


def decode_each(tokenizer, values: list[int], limit: int) -> list[str]:
    return [tokenizer.decode([x]) for x in values[:limit]]


def describe_case(
    name: str,
    tokenizer,
    token_ids: list[int],
    prefix_token_id: int,
    max_seq_len: int,
    display_limit: int,
) -> None:
    context, continuation, engine_input = first_disjoint_rolling_window(
        token_ids,
        prefix_token_id,
        max_seq_len,
    )
    ctxlen = len(context)
    scored_tokens = engine_input[ctxlen:]

    print(f"\n== {name} ==")
    print(f"tok_encode ids: {compact(token_ids, display_limit)}")
    print(f"tok_encode decoded: {decode_each(tokenizer, token_ids, display_limit)}")
    print(f"rolling context ids: {compact(context, display_limit)}")
    print(f"rolling continuation ids: {compact(continuation, display_limit)}")
    print(f"engine input ids: {compact(engine_input, display_limit)}")
    print(f"engine input decoded: {decode_each(tokenizer, engine_input, display_limit)}")
    print(f"ctxlen: {ctxlen}")
    print(f"scored token ids: {compact(scored_tokens, display_limit)}")
    print(f"scored decoded: {decode_each(tokenizer, scored_tokens, display_limit)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenizer-only reproduction of lm-eval SGLang/vLLM tok_encode "
            "plus loglikelihood_rolling prefix behavior."
        )
    )
    parser.add_argument(
        "--model",
        default="/data/like/hf-models/llama3.1-70B",
        help="HF tokenizer/model path.",
    )
    parser.add_argument(
        "--text",
        default="Hello world",
        help="Text to encode.",
    )
    parser.add_argument(
        "--sglang-add-bos-token",
        type=parse_bool,
        default=False,
        help="Controls SGLangLM.add_bos_token in the simulated SGLang adapter.",
    )
    parser.add_argument(
        "--vllm-add-bos-token",
        type=parse_optional_bool,
        default=None,
        help="Controls VLLM.add_bos_token. Use none to match current vLLM default.",
    )
    parser.add_argument(
        "--vllm-add-special-tokens",
        type=parse_optional_bool,
        default=None,
        help="Controls explicit add_special_tokens passed to vLLM tok_encode.",
    )
    parser.add_argument(
        "--prefix-token-id",
        type=int,
        default=None,
        help="Override rolling prefix token id. Defaults to tokenizer BOS, then EOS.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=131071,
        help="max_seq_len used by get_rolling_token_windows.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.",
    )
    parser.add_argument(
        "--display-limit",
        type=int,
        default=24,
        help="Maximum tokens to print per decoded list.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    prefix_token_id = args.prefix_token_id
    if prefix_token_id is None:
        prefix_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )

    print(f"model: {args.model}")
    print(f"text: {args.text!r}")
    print(f"tokenizer: {type(tokenizer)}")
    print(f"bos_token_id: {tokenizer.bos_token_id}, bos_token: {tokenizer.bos_token!r}")
    print(f"eos_token_id: {tokenizer.eos_token_id}, eos_token: {tokenizer.eos_token!r}")
    print(f"tokenizer.add_bos_token: {getattr(tokenizer, 'add_bos_token', None)!r}")
    print(
        f"default tokenizer(text).input_ids: "
        f"{compact(tokenizer(args.text).input_ids, args.display_limit)}"
    )
    print(f"prefix_token_id: {prefix_token_id}, decoded: {tokenizer.decode([prefix_token_id])!r}")

    sglang_tokens = sglang_tok_encode(
        tokenizer,
        args.text,
        add_bos_token=args.sglang_add_bos_token,
    )
    vllm_tokens = vllm_tok_encode(
        tokenizer,
        args.text,
        prefix_token_id=prefix_token_id,
        add_bos_token=args.vllm_add_bos_token,
        add_special_tokens=args.vllm_add_special_tokens,
    )

    describe_case(
        f"sglang simulated, add_bos_token={args.sglang_add_bos_token}",
        tokenizer,
        sglang_tokens,
        prefix_token_id,
        args.max_seq_len,
        args.display_limit,
    )
    describe_case(
        (
            "vllm simulated, "
            f"add_bos_token={args.vllm_add_bos_token}, "
            f"add_special_tokens={args.vllm_add_special_tokens}"
        ),
        tokenizer,
        vllm_tokens,
        prefix_token_id,
        args.max_seq_len,
        args.display_limit,
    )


if __name__ == "__main__":
    main()
