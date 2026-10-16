
from __future__ import annotations

import re
from typing import List, Tuple


_THINK_RE = re.compile(r"<(think|reason)>.*?</\1>", flags=re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", flags=re.IGNORECASE | re.DOTALL)


def _parse_action(text: str) -> Tuple[str, bool]:
    pass
    value = str(text)
    matches = _ACTION_RE.findall(value)
    if _THINK_RE.search(value) is None or len(matches) != 1:
        return "", False

    action = re.sub(r"\s+", " ", matches[0]).strip()
    if not action or len(action) > 512 or "<" in action or ">" in action:
        return "", False
    return action, True


def scienceworld_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    pass





    projected = []
    valid = []
    for text in actions:
        action, is_valid = _parse_action(str(text))
        projected.append(action)
        valid.append(int(is_valid))
    return projected, valid
