"""
Shared schema definitions for dataset columns used across scripts.
"""

VIDEO_COLUMN = "video"
TARGET_COLUMN = "trick"
CONTEXT_COLUMNS = ["key_foot", "person"]

# Columns expected in every exported row, in canonical order.
EXPECTED_COLUMNS = [VIDEO_COLUMN, TARGET_COLUMN, *CONTEXT_COLUMNS]

# Categorical columns for vocab metadata / label stats (excludes video paths).
CATEGORICAL_COLUMNS = [TARGET_COLUMN, *CONTEXT_COLUMNS]
LABEL_COLUMNS = CATEGORICAL_COLUMNS
