# Using the RBN Analytics Tool with RBN Log Data

## RBN Log Format

The tool is optimized for RBN log files with whitespace-separated columns:

```
callsign de_pfx de_cont freq band dx dx_pfx dx_cont mode db date speed tx_mode
IK3STG   I      EU      3530.5 80m SQ2AAA SP EU CQ 25 11/6/2025 0:00 29 CW
WZ7I     K      NA      14100  20m OA4B   OA SA NCDXF B 8 11/6/2025 0:00 20 CW
```

### Column Mapping

The tool automatically maps RBN columns to internal format:

| RBN Column | Internal Name | Description |
|------------|---------------|-------------|
| `callsign` | `skimmer` | Skimmer station ID |
| `freq` | `freq_khz` | Frequency in kHz (converted to Hz) |
| `band` | `band` | Band (e.g., "20m", "80m") |
| `dx` | `dx_call` | Decoded callsign |
| `mode` | `mode` | Mode (CW, RTTY, SSB, etc.) |
| `db` | `snr_db` | Signal-to-noise ratio in dB |
| `date` | `timestamp` | Timestamp (format: "11/6/2025 0:00") |

**Unused columns** (automatically dropped for performance):
- `de_pfx`, `de_cont`, `dx_pfx`, `dx_cont`, `speed`, `tx_mode`

## Quick Start with RBN Data

### 1. Prepare Your RBN Log Files

Your RBN log files should be whitespace-separated text files with the columns shown above. 

**No preprocessing needed!** The tool automatically extracts compressed files.

**Supported formats:**
- ✅ Plain text: `.txt`, `.csv`
- ✅ Compressed: `.gz`, `.bz2`
- ✅ Archives: `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tgz`

Example file structure:
```
data/
├── cqww_wpx_cw_2023.txt.gz      ← Automatically extracted
├── cqww_wpx_cw_2024.zip         ← Automatically extracted
├── cqww_dx_cw_2024.tar.gz       ← Automatically extracted
└── other_contest.txt            ← Used directly
```

### 2. Run the Analysis

**Process a single RBN log file:**
```bash
rbn-train --input data/cqww_wpx_cw_2024.txt --output-dir results/
```

**Process all RBN logs in a directory (including archives):**
```bash
rbn-train --input data/ --output-dir results/
```

**Process specific contest logs:**
```bash
rbn-train --input "data/cqww_*.txt" --output-dir results/
```

**Process compressed files directly:**
```bash
rbn-train --input "data/*.gz" --output-dir results/
```

**Mix of compressed and uncompressed:**
```bash
rbn-train --input data/ --output-dir results/
# Automatically handles .txt, .csv, .gz, .bz2, .zip, .tar.gz, etc.
```

### 3. Performance Optimizations

The tool includes several optimizations for large RBN datasets:

✅ **Automatic Archive Extraction**: Handles .zip, .gz, .bz2, .tar, .tar.gz files  
✅ **Column Selection**: Only loads 7 needed columns (drops 5 unused columns)  
✅ **Chunked Reading**: Processes 100k rows at a time to manage memory  
✅ **Optimized Data Types**: Uses efficient string types for text columns  
✅ **Whitespace Parsing**: Handles RBN's space-separated format natively  

### 4. Expected Performance

For typical RBN contest data:
- **10M spots**: ~5-10 minutes on modern hardware
- **Memory usage**: ~2-4 GB peak (with chunking)
- **Output size**: confusion_model.json (~1-5 MB), priors.json (~100 KB - 1 MB)

## Configuration for RBN Data

Create a `rbn_config.json` for RBN-specific settings:

```json
{
  "cluster_time_seconds": 60,
  "cluster_freq_bin_hz": 500,
  "min_cluster_skimmers": 4,
  "min_cluster_share_percent": 70.0,
  "stability_freq_bin_hz": 1000,
  "stability_min_clusters": 5,
  "stability_min_share_percent": 80.0,
  "modes": ["CW", "RTTY", "SSB"],
  "min_snr_db": -999.0,
  "max_call_length": 16,
  "min_call_length": 2
}
```

Then run:
```bash
rbn-train --input data/ --output-dir results/ --config rbn_config.json
```

## Example Workflow

```bash
# 1. Install the tool
pip install -e .

# 2. Download RBN logs for CQWW WPX CW and CQWW DX CW
# (Place them in a data/ directory)

# 3. Run the analysis
rbn-train --input data/ --output-dir output/

# 4. Check the results
ls output/
# confusion_model.json  priors.json

# 5. View statistics
cat output/priors.json | jq '.calls | length'
# Shows number of unique callsigns found
```

## Output Files

### confusion_model.json
Contains character-level error statistics by mode and SNR band:
- Substitution patterns (e.g., "L" → "I", "0" → "O")
- Deletion patterns (missing characters)
- Insertion patterns (extra characters)

### priors.json
Contains global callsign frequencies:
```json
{
  "calls": {
    "K3LR": 1245,
    "W3LPL": 987,
    "KC1XX": 756,
    ...
  }
}
```

## Troubleshooting

**"Missing required columns" error?**
- Verify your file has the standard RBN format with whitespace separation
- Check that column names match: `callsign freq band dx mode db date`

**Low number of spots loaded?**
- Check the mode filter (default: CW, RTTY, SSB)
- Verify SNR threshold (default: -999 dB, accepts all)
- Check callsign length limits (default: 2-16 characters)

**Out of memory?**
- The tool uses chunked reading (100k rows at a time)
- If still having issues, process files individually instead of all at once

**Slow processing?**
- The tool is already optimized to drop unused columns
- Consider filtering by mode if you only need CW data
- Use SSD storage for faster I/O

## Tips for Best Results

1. **Use 2-3 years of data**: Combine multiple contests for better statistics
2. **Include both WPX and DX contests**: More diverse callsigns
3. **CW mode recommended**: Most accurate decoding, best for training
4. **High-activity periods**: Contest weekends provide most data
5. **Multiple skimmers**: More skimmers = better consensus

## Next Steps

After generating the confusion model and priors:
1. Load them into your Go runtime correction engine
2. Use the confusion model for error probability estimation
3. Use priors for Bayesian callsign correction
4. Re-run analysis periodically (e.g., after each major contest)
