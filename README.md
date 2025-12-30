# RBN Callsign Correction Analytics Tool

An offline Python tool that analyzes large volumes of RBN (Reverse Beacon Network) spot data from major CW/RTTY contests to infer true callsigns and learn decoder error patterns.

## Quick Start

**Most common scenario:** You have a directory of RBN contest zip files downloaded from the RBN archive.

```bash
# 1. Install the tool
pip install -e .

# 2. Run analysis on your directory of zip files (auto-detect CPU cores)
rbn-train --input /path/to/rbn_logs/ --output-dir output/ --workers 0
```

That's it. The tool will:
- Automatically extract all `.zip` files in the directory
- Process all contest data (CW and RTTY spots)
- Generate output files in the `output/` directory

**Processing time:** Expect ~1 hour per million spots on a modern multi-core machine.

## Installation

```bash
# Clone the repository
git clone https://github.com/N2WQ/rbn-spot-analytics.git
cd rbn-spot-analytics

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0

## Input Data

### Supported Formats

The tool accepts RBN contest data in these formats:
- **ZIP archives** (most common) - automatically extracted
- **CSV files** - whitespace or comma-separated
- **Directories** - scans for all zip/csv files

### Where to Get RBN Data

Download contest spot data from the RBN archive at https://www.reversebeacon.net/raw_data/

### Data Format

RBN files contain whitespace-separated columns including:
- `callsign` - Skimmer station that heard the signal
- `dx` - Decoded callsign (what we're analyzing)
- `freq` - Frequency in kHz
- `band` - Amateur band (20m, 40m, etc.)
- `tx_mode` - Transmission mode (CW, RTTY)
- `db` - Signal-to-noise ratio

## Command Reference

### Basic Syntax

```bash
rbn-train --input <INPUT> --output-dir <OUTPUT> [--workers N] [--config FILE]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Directory of zip files, single file, or glob pattern |
| `--output-dir` | Yes | Directory for output files (created if doesn't exist) |
| `--workers` | No | Number of CPU cores to use (default: 1, use 0 for auto-detect) |
| `--config` | No | Path to JSON configuration file |

### Examples

```bash
# Process all zip files in a directory (RECOMMENDED)
rbn-train --input /path/to/rbn_logs/ --output-dir output/ --workers 0

# Process a single zip file
rbn-train --input cqww_cw_2024.zip --output-dir output/ --workers 0

# Process specific files with glob pattern
rbn-train --input "rbn_logs/cqww_*.zip" --output-dir output/ --workers 0

# Use specific number of workers (e.g., 4 cores)
rbn-train --input rbn_logs/ --output-dir output/ --workers 4

# Sequential processing (single core, slower but uses less memory)
rbn-train --input rbn_logs/ --output-dir output/ --workers 1
```

## Parallel Processing

### Recommended Settings

| System | Recommended `--workers` |
|--------|-------------------------|
| Any system | `0` (auto-detect) |
| 4-core CPU | `3` or `4` |
| 8-core CPU | `6` or `7` |
| Low memory (<8GB RAM) | `1` or `2` |

### What Gets Parallelized

- Cluster consensus computation
- Stability-based truth refinement
- Confusion model building
- Spotter reliability calculation

### Performance Tips

- Use `--workers 0` to automatically use (CPU cores - 1)
- Data loading is sequential (I/O bound), parallelization helps with analysis
- Memory usage scales with worker count (~500MB per worker)
- For very large datasets (50M+ spots), use fewer workers to avoid memory issues

## Output Files

After processing, you'll find these files in your output directory:

### Core Outputs

| File | Description |
|------|-------------|
| `confusion_model.json` | Character-level decoder error patterns by mode and SNR |
| `priors.json` | Callsign frequency counts (how often each call appears) |

### Spotter Reliability

| File | Description |
|------|-------------|
| `spotter_reliability.json` | Full reliability data with band/mode breakdowns |
| `spotter_reliability.txt` | Simple format: `SKIMMER RELIABILITY` |
| `spotter_reliability_cw.txt` | CW-only reliability scores |
| `spotter_reliability_rtty.txt` | RTTY-only reliability scores |

### Band/Segment Priors

| File | Description |
|------|-------------|
| `band_priors.json` | Callsign counts per amateur band |
| `segment_priors.json` | Callsign counts per frequency segment |
| `call_quality_priors.txt` | Simple format for Go integration |

## Configuration (Optional)

For most users, the defaults work well. Create `config.json` only if you need to customize:

```json
{
  "modes": ["CW", "RTTY"],
  "min_spotter_spots": 100,
  "workers": 0
}
```

### Key Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `modes` | `["CW", "RTTY"]` | Which transmission modes to analyze |
| `min_spotter_spots` | `1000` | Minimum spots for skimmer to be included in reliability |
| `min_cluster_skimmers` | `4` | Minimum skimmers to establish cluster consensus |
| `min_cluster_share_percent` | `70.0` | Percentage agreement needed for provisional truth |

## Troubleshooting

### "No files found"
- Check that your input path is correct
- Ensure the directory contains `.zip` or `.csv` files

### Out of memory
- Reduce `--workers` to `1` or `2`
- Process fewer files at a time

### Processing is slow
- Use `--workers 0` to enable parallel processing
- Ensure you're not running on a single core (`--workers 1` is default)

### Missing mode data
- RBN uses `tx_mode` column for CW/RTTY (not the `mode` column)
- The tool handles this automatically

## Go Runtime Integration

The simple text formats are designed for direct use with Go's correction system:

**spotter_reliability_cw.txt** / **spotter_reliability_rtty.txt**:
```
W3LPL 0.97
N0RZA 0.90
K1TTT 0.85
```

**call_quality_priors.txt**:
```
K3LR 5
W1XYZ 4
N0RZA 3
```

## Features

- Analyzes millions of RBN spots from contest data
- Infers true callsigns via clustering and stability analysis
- Learns character-level decoder error patterns (substitutions, insertions, deletions)
- Generates confusion models and global callsign priors for runtime correction engines
- **Spotter reliability analysis** - per-skimmer accuracy metrics by band and mode
- **Band-specific priors** - callsign frequency counts per amateur band
- **Frequency segment priors** - callsign counts for CW/RTTY segments within bands
- **Parallel processing** - multi-core support for faster analysis of large datasets
- Memory-efficient processing with chunked CSV reading
- Automatic zip file extraction

## Testing

```bash
pytest tests/
```

## License

MIT
