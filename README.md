# ProtLake

An efficient storage format for large datasets of protein structures and associated metadata using DeltaLake and PyArrow.

## Overview

ProtLake provides a scalable solution for storing and querying protein structure datasets, particularly designed for AlphaFold3 outputs. It combines compressed binary CIF (BinaryCIF) format for structures with Delta Lake for metadata management, enabling efficient storage and fast queries on large protein datasets.

## Key Features

- **Efficient Storage**: Uses BinaryCIF compression for protein structures and custom PACK format for sharding
- **Metadata Management**: Stores confidence scores, ranking metrics, and other metadata in queryable Delta Lake tables  
- **Scalable Architecture**: Handles large datasets with automatic sharding and optimization
- **Fast Queries**: Arrow-based querying for rapid data access and filtering
- **AF3 Integration**: Native support for AlphaFold3 output formats

## Installation

Create the conda environment:

```bash
conda env create -f mamba_env.yml
conda activate af3
```

## Quick Start

### Writing Data

```python
from write import IngestConfig, AF3IngestPipeline

# Configure ingestion
cfg = IngestConfig(
    out_path="path/to/protlake",
    batch_size_metadata=2500,
    verbose=True
)

# Run ingestion pipeline
input_dirs = ["path/to/af3/output1", "path/to/af3/output2"]
AF3IngestPipeline(cfg).run(input_dirs)
```

### Reading Data

```python
from deltalake import DeltaTable
from read import pread_bcif_to_atom_array, pread_json_msgpack_to_dict

# Load metadata table
dt = DeltaTable("path/to/protlake/delta")
df = dt.to_pandas()

# Get specific structure
row = df[df["name"] == "protein_name"].iloc[0]
atom_array = pread_bcif_to_atom_array(
    f"path/to/protlake/shards/{row['bcif_shard']}", 
    row["bcif_off"], 
    row["bcif_len"]
)
```

## Project Structure

- `write.py` - Data ingestion pipeline for AF3 outputs
- `read.py` - Functions for reading structures and metadata  
- `utils.py` - Delta Lake maintenance and utilities
- `tests/af3/` - Test data and integration examples
- `analyze/` - Analysis tools and scripts
