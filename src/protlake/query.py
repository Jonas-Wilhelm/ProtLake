import os
from typing import Any, Dict

import pyarrow.dataset as ds
from deltalake import DeltaTable

from protlake.write.core import RetryConfig, load_delta_table_with_retries


def check_exists(delta_path: str, keys: Dict[str, Any], retry_config: RetryConfig) -> bool:
    """
    Check if an entry matching the given key values exists in a Delta table.

    Args:
        delta_path: Path to the Delta table directory.
        keys: Dictionary of column-value pairs to match.
        retry_config: Retry settings used when loading the Delta table.

    Returns:
        True if at least one matching entry exists.
    """
    if not keys:
        raise ValueError("keys dict must not be empty")

    if not DeltaTable.is_deltatable(f"file://{os.path.abspath(delta_path)}"):
        return False

    filter_expr = None
    for col, val in keys.items():
        cond = ds.field(col) == val
        filter_expr = cond if filter_expr is None else (filter_expr & cond)

    dt = load_delta_table_with_retries(
        delta_path=delta_path,
        retry_config=retry_config,
    )

    scanner = dt.to_pyarrow_dataset().scanner(filter=filter_expr, columns=["id_hex"])
    for batch in scanner.to_batches():
        if batch.num_rows > 0:
            return True
    return False
