# RBN Callsign Correction Analytics Tool

An offline Python tool for analyzing RBN (Reverse Beacon Network) spot data to infer true callsigns and learn decoder error patterns.

## Overview

This tool processes large volumes of RBN spot data from major CW contests to:

1. Infer high-confidence "true" callsigns via clustering and stability analysis
2. Learn empirically how decoders misread callsigns as a function of mode and SNR
3. Produce confusion models and callsign priors for runtime call-correction engines

## Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

```bash
rbn-train --input spots.csv --output-dir output/
```

### With Compressed Files (Automatic Extraction)

```bash
# Automatically extracts .gz, .bz2, .zip, .tar.gz, etc.
rbn-train --input spots.txt.gz --output-dir output/
```

### With Directory Input

```bash
# Processes all files including archives
rbn-train --input data/ --output-dir output/
```

### With Glob Pattern

```bash
rbn-train --input "data/cqww_*.csv" --output-dir output/
```

### With Configuration File

```bash
rbn-train --input spots.csv --output-dir output/ --config config.json
```

## Configuration

Create a JSON configuration file to customize analysis parameters:

```json
{
  "cluster_time_seconds": 60,
  "cluster_freq_bin_hz": 500,
  "min_cluster_skimmers": 4,
  "min_cluster_share_percent": 70.0,
  "stability_freq_bin_hz": 1000,
  "stability_min_clusters": 5,
  "stability_min_share_percent": 80.0,
  "snr_bands": [-999.0, 0.0, 6.0, 12.0, 18.0, 24.0, 999.0],
  "modes": ["CW", "RTTY", "SSB"],
  "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/",
  "unknown_char": "?",
  "min_snr_db": -999.0,
  "max_call_length": 16,
  "min_call_length": 2
}
```

## Input CSV Format

CSV files must contain the following columns:

- `timestamp`: UTC timestamp (ISO8601 format)
- `band`: Band (e.g., "20m", "40m")
- `mode`: Mode (e.g., "CW", "RTTY", "SSB")
- `skimmer`: Skimmer station ID
- `dx_call`: Decoded callsign
- `snr_db`: Signal-to-noise ratio in dB
- `freq_hz` or `freq_khz`: Frequency

## Output Files

The tool produces two JSON files:

### confusion_model.json

Contains character-level error statistics:
- Substitution counts (4D: mode × SNR band × true char × observed char)
- Deletion counts (3D: mode × SNR band × true char)
- Insertion counts (3D: mode × SNR band × observed char)

### priors.json

Contains global callsign frequency counts:
- Mapping of normalized callsigns to occurrence counts

## Algorithm

The tool operates in three main passes:

1. **Pass A - Micro-clustering**: Groups spots by time/frequency proximity and determines provisional true callsigns via skimmer consensus

2. **Pass B - Stability Analysis**: Validates provisional callsigns across larger frequency bins to improve accuracy

3. **Pass C - Confusion & Priors**: Builds character-level error statistics and global callsign frequency counts

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=rbn_train --cov-report=html
```

## Development

The package structure:

```
rbn_train/
├── __init__.py
├── cli.py              # CLI orchestration
├── config.py           # Configuration management
├── io.py               # CSV loading
├── types.py            # Data structures
├── clustering.py       # Micro-clustering
├── stability.py        # Stability analysis
├── confusion.py        # Confusion model building
└── priors.py           # Global priors
```

## License

See LICENSE file for details.
