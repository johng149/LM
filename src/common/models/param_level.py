from enum import Enum

class ParamLevel(Enum):
    # can be:
    # 1. Required: must be provided (causes error if not provided)
    # 2. Suggested: should be provided (causes warning if not provided)
    # 3. Optional: can be provided (no issue if not provided)

    REQUIRED = 1
    SUGGESTED = 2
    OPTIONAL = 3