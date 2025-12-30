# Quick Start Guide

Get started with the RBN Analytics Tool in 3 simple steps!

## Installation

```bash
pip install -e .
```

## Usage

### Option 1: Compressed RBN Logs (Most Common)

```bash
# Your downloaded RBN logs are compressed? No problem!
rbn-train --input rbn_logs.tar.gz --output-dir results/
```

### Option 2: Directory of Files

```bash
# Mix of compressed and uncompressed files
rbn-train --input data/ --output-dir results/
```

### Option 3: Specific Pattern

```bash
# Process specific contests
rbn-train --input "data/cqww_*.gz" --output-dir results/
```

## What You Get

Two JSON files in the output directory:

1. **`confusion_model.json`** - Character-level error patterns
2. **`priors.json`** - Callsign frequency counts

## Supported File Formats

✅ Plain text: `.txt`, `.csv`  
✅ Compressed: `.gz`, `.bz2`  
✅ Archives: `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`

**No manual extraction needed!**

## Example Workflow

```bash
# 1. Install
pip install -e .

# 2. Download RBN logs (they're usually compressed)
# Example: cqww_wpx_2024.txt.gz

# 3. Run analysis (no need to extract!)
rbn-train --input cqww_wpx_2024.txt.gz --output-dir results/

# 4. Check results
ls results/
# confusion_model.json  priors.json

# 5. Use in your Go correction engine
```

## Performance

- **10M spots**: ~5-10 minutes
- **Memory**: ~2-4 GB peak
- **Automatic**: Extracts archives, drops unused columns, chunks processing

## Need Help?

```bash
rbn-train --help
```

## Documentation

- **Full Usage Guide**: See `RBN_USAGE_GUIDE.md`
- **Archive Support**: See `ARCHIVE_SUPPORT.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`

## Common Issues

**"No files found"**
- Check file extensions (`.txt`, `.csv`, or archives)
- Verify RBN log format (whitespace-separated columns)

**"Missing required columns"**
- Ensure logs have: `callsign freq band dx mode db date`

**"Out of memory"**
- Tool uses chunked reading, should handle large files
- Try processing files individually if needed

## That's It!

The tool handles everything automatically:
- ✅ Extracts archives
- ✅ Parses RBN format
- ✅ Filters and normalizes data
- ✅ Builds confusion model
- ✅ Computes priors
- ✅ Exports JSON

Just point it at your RBN logs and go!
