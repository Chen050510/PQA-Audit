"""Tokenizer-aware generation stop resolution for E31."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _ordered_unique_ids(values: Iterable[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        token_id = int(value)
        if token_id not in result:
            result.append(token_id)
    return result


def _base_eos_ids(tokenizer: Any) -> list[int]:
    value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("tokenizer has no eos_token_id")
    if isinstance(value, int):
        return [int(value)]
    return _ordered_unique_ids(value)


def _special_registration_source(tokenizer: Any, token_id: int) -> str | None:
    """Accept both tokenizer special maps and AddedToken(special=True)."""
    if token_id in {int(value) for value in tokenizer.all_special_ids}:
        return "all_special_ids"
    decoder = getattr(tokenizer, "added_tokens_decoder", {})
    added = decoder.get(token_id) if hasattr(decoder, "get") else None
    if added is None and hasattr(decoder, "get"):
        added = decoder.get(str(token_id))
    if added is not None and getattr(added, "special", False) is True:
        return "added_tokens_decoder.special"
    return None


def resolve_generation_stops(
    tokenizer: Any,
    additional_eos_tokens: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Resolve configured stop strings and fail closed on ambiguous tokens."""

    base_ids = _base_eos_ids(tokenizer)
    token_strings = list(additional_eos_tokens or [])
    if len(token_strings) != len(set(token_strings)):
        raise ValueError("additional_eos_tokens contains duplicates")

    unknown_id = tokenizer.unk_token_id
    resolved: list[dict[str, Any]] = []
    for token in token_strings:
        if not isinstance(token, str) or not token:
            raise ValueError("additional eos token must be a nonempty string")
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (unknown_id is not None and int(token_id) == int(unknown_id)):
            raise ValueError(f"additional eos token is unknown: {token!r}")
        token_id = int(token_id)
        encoded = [int(value) for value in tokenizer.encode(token, add_special_tokens=False)]
        if encoded != [token_id]:
            raise ValueError(
                f"additional eos token must encode to one exact token: {token!r} -> {encoded}"
            )
        registration_source = _special_registration_source(tokenizer, token_id)
        if registration_source is None:
            raise ValueError(f"additional eos token is not registered special: {token!r}")
        visible = tokenizer.decode([token_id], skip_special_tokens=True)
        if visible:
            raise ValueError(
                f"additional eos token remains visible after special-token decoding: {token!r}"
            )
        resolved.append({
            "token": token,
            "token_id": token_id,
            "registration_source": registration_source,
        })

    all_ids = _ordered_unique_ids([*base_ids, *(row["token_id"] for row in resolved)])
    generation_value: int | list[int] = all_ids[0] if len(all_ids) == 1 else all_ids
    return {
        "base_eos_token_ids": base_ids,
        "additional_eos_tokens": resolved,
        "additional_eos_token_ids": [row["token_id"] for row in resolved],
        "all_eos_token_ids": all_ids,
        "generation_eos_token_id": generation_value,
        "resolution_policy": "single_registered_special_invisible_token_only",
    }
