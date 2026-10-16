
import re
from typing import List, Tuple




ACTION_TO_INDEX = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
    "northeast": 4,
    "southeast": 5,
    "southwest": 6,
    "northwest": 7,
}

ALIASES = {
    "n": "north",
    "up": "north",
    "e": "east",
    "right": "east",
    "s": "south",
    "down": "south",
    "w": "west",
    "left": "west",
    "ne": "northeast",
    "north east": "northeast",
    "up right": "northeast",
    "se": "southeast",
    "south east": "southeast",
    "down right": "southeast",
    "sw": "southwest",
    "south west": "southwest",
    "down left": "southwest",
    "nw": "northwest",
    "north west": "northwest",
    "up left": "northwest",
}


def _parse_action(text: str):
    matches = list(
        re.finditer(
            r"<answer>\s*(.*?)\s*</answer>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        return -1, False

    action = matches[-1].group(1).strip().lower()
    action = re.sub(r"[-_]+", " ", action)
    action = re.sub(r"\s+", " ", action)
    action = ALIASES.get(action, action.replace(" ", ""))
    if action not in ACTION_TO_INDEX:
        return -1, False
    return ACTION_TO_INDEX[action], True


def minihack_room_projection(actions: List[str]) -> Tuple[List[int], List[int]]:
    pass
    projected = []
    valid = []
    for text in actions:
        action, is_valid = _parse_action(str(text))
        projected.append(action)
        valid.append(int(is_valid))
    return projected, valid
