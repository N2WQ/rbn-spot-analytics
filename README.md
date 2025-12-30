# RBN Callsign Correction Analytics Tool

An offline Python tool that analyzes large volumes of RBN (Reverse Beacon Network) spot data from major CW/RTTY contests to infer true callsigns and learn decoder error patterns.

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
- Outputs both JSON (for analysis) and simple text formats (for Go runtime integration)

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Single file
rbn-train --input data/cqww_2023.csv --output-dir output/

# Directory
rbn-train --input data/ --output-dir output/

# Glob pattern
rbn-train --input "data/cqww_*.csv" --output-dir output/

# With config file
rbn-train --input data/ --output-dir output/ --config config.json

# Enable parallel processing with 4 workers
rbn-train --input data/ --output-dir output/ --workers 4

# Auto-detect worker count (uses CPU count - 1)
rbn-train --input data/ --output-dir output/ --workers 0
```

## Output Files

### Core Outputs

| File | Format | Description |
|------|--------|-------------|
| `confusion_model.json` | JSON | Character-level error statistics by mode and SNR band |
| `priors.json` | JSON | Global callsign frequency counts |

### Extended Outputs

| File | Format | Description |
|------|--------|-------------|
| `spotter_reliability.json` | JSON | Per-skimmer accuracy metrics with band/mode breakdown |
| `spotter_reliability.txt` | Text | Simple `SKIMMER RELIABILITY` format for Go integration |
| `band_priors.json` | JSON | Callsign counts per amateur band |
| `call_quality_priors.txt` | Text | Simple `CALL SCORE [FREQ_KHZ]` format for Go integration |
| `segment_priors.json` | JSON | Callsign counts per frequency segment (CW/RTTY) |

### Output Structure Examples

**spotter_reliability.json:**
```json
{
  "min_spots_threshold": 100,
  "skimmers_total": 500,
  "skimmers_included": 350,
  "skimmers": {
    "W3LPL": {"total": 50000, "exact": 48500, "reliability": 0.97},
    "N0RZA": {"total": 12000, "exact": 10800, "reliability": 0.90}
  },
  "by_band": {
    "20m": {"W3LPL": {"total": 15000, "exact": 14800, "reliability": 0.987}}
  },
  "by_mode": {
    "CW": {"W3LPL": {"total": 45000, "exact": 44000, "reliability": 0.978}}
  }
}
```

**band_priors.json:**
```json
{
  "bands": ["160m", "80m", "40m", "20m", "15m", "10m"],
  "global": {"K3LR": 1500, "W1XYZ": 800},
  "by_band": {
    "20m": {"K3LR": 500, "W1XYZ": 200},
    "40m": {"K3LR": 300, "W1XYZ": 400}
  }
}
```

## Configuration

Create a `config.json` file to customize analysis parameters:

```json
{
  "cluster_time_seconds": 60,
  "cluster_freq_bin_hz": 500,
  "min_cluster_skimmers": 4,
  "min_cluster_share_percent": 70.0,
  "stability_freq_bin_hz": 1000,
  "stability_min_clusters": 5,
  "stability_min_share_percent": 80.0,
  "modes": ["CW", "RTTY"],
  "snr_bands": [-999.0, 0.0, 6.0, 12.0, 18.0, 24.0, 999.0],
  "output_spotter_reliability": true,
  "output_band_priors": true,
  "output_segment_priors": true,
  "min_spotter_spots": 100,
  "output_simple_formats": true,
  "workers": 1,
  "parallel_chunk_size": 1000
}
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_spotter_reliability` | `true` | Generate spotter reliability analysis |
| `output_band_priors` | `true` | Generate band-specific priors |
| `output_segment_priors` | `true` | Generate frequency segment priors |
| `min_spotter_spots` | `100` | Minimum spots for skimmer to be included in reliability |
| `output_simple_formats` | `true` | Generate Go-compatible text formats |
| `workers` | `1` | Number of parallel workers (0=auto-detect, 1=sequential) |
| `parallel_chunk_size` | `1000` | Number of clusters per parallel batch |

## Go Runtime Integration

The simple text formats are designed for direct use with Go's correction system:

**spotter_reliability.txt** - Load with `LoadSpotterReliability()`:
```
W3LPL 0.97
N0RZA 0.90
K1TTT 0.85
```

**call_quality_priors.txt** - Load with `LoadCallQualityPriors()`:
```
K3LR 5
W1XYZ 4
N0RZA 3 14050
```

## Parallel Processing

The tool supports parallel processing to speed up analysis of large datasets. When enabled, the following operations run in parallel across multiple CPU cores:

- **Provisional truth labeling** - Cluster consensus computation
- **Truth refinement** - Stability-based callsign correction
- **Confusion model building** - Character error pattern extraction
- **Spotter reliability** - Per-skimmer accuracy computation

### Usage

```bash
# Use 4 parallel workers
rbn-train --input data/ --output-dir output/ --workers 4

# Auto-detect optimal worker count (CPU cores - 1)
rbn-train --input data/ --output-dir output/ --workers 0
```

### Performance Notes

- Parallelization provides the most benefit with large datasets (100K+ spots)
- On a 4-core machine, expect 2-3x speedup for compute-intensive operations
- Data loading remains sequential (I/O bound, not CPU bound)
- Memory usage scales with worker count due to process forking

### Configuration via JSON

```json
{
  "workers": 4,
  "parallel_chunk_size": 1000
}
```

- `workers`: Number of worker processes (default: 1 = sequential)
- `parallel_chunk_size`: Clusters per batch (larger = less overhead, more memory)

## Requirements

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0

## Testing

```bash
pytest tests/
```

## License

MIT
