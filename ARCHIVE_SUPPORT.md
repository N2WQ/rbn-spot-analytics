# Automatic Archive Extraction

The RBN Analytics Tool automatically extracts compressed files, so you don't need to manually unzip your RBN logs!

## Supported Archive Formats

| Format | Extension | Example |
|--------|-----------|---------|
| Gzip | `.gz` | `cqww_2024.txt.gz` |
| Bzip2 | `.bz2` | `cqww_2024.txt.bz2` |
| Zip | `.zip` | `rbn_logs.zip` |
| Tar | `.tar` | `logs.tar` |
| Tar+Gzip | `.tar.gz`, `.tgz` | `rbn_2024.tar.gz` |
| Tar+Bzip2 | `.tar.bz2` | `rbn_2024.tar.bz2` |

## How It Works

### Single Archive File

```bash
# Automatically extracts and processes
rbn-train --input cqww_2024.txt.gz --output-dir results/
```

The tool will:
1. Detect the `.gz` extension
2. Extract to a temporary directory
3. Process the extracted file(s)
4. Clean up temporary files when done

### Directory with Mixed Files

```bash
rbn-train --input data/ --output-dir results/
```

Given this directory structure:
```
data/
├── contest1.txt          ← Processed directly
├── contest2.txt.gz       ← Extracted, then processed
├── contest3.zip          ← Extracted, then processed
└── contest4.tar.gz       ← Extracted, then processed
```

The tool automatically:
- Processes plain `.txt` and `.csv` files directly
- Extracts all archives (`.gz`, `.zip`, `.tar.gz`, etc.)
- Processes all extracted files
- Handles nested archives (archives within archives)

### Glob Patterns with Archives

```bash
# Process all gzipped files
rbn-train --input "data/*.gz" --output-dir results/

# Process all archives
rbn-train --input "data/*.{gz,zip,bz2}" --output-dir results/
```

## Examples

### Example 1: Downloaded RBN Logs (Compressed)

```bash
# You downloaded: cqww_wpx_2023.txt.gz, cqww_wpx_2024.txt.gz
# No need to extract manually!

rbn-train --input "*.gz" --output-dir results/
```

### Example 2: Mixed Archive Types

```bash
data/
├── 2023_cqww_wpx.txt.gz
├── 2024_cqww_wpx.zip
├── 2024_cqww_dx.tar.gz
└── 2025_arrl_dx.txt

# Process everything at once
rbn-train --input data/ --output-dir results/
```

### Example 3: Large Tar Archive

```bash
# Single tar.gz containing multiple contest logs
rbn-train --input all_contests_2024.tar.gz --output-dir results/
```

The tool will:
1. Extract the tar.gz
2. Find all `.txt` and `.csv` files inside
3. Process each one
4. Combine results into single confusion model and priors

## Performance Notes

### Extraction Speed

- **Gzip (`.gz`)**: Fast, single-threaded
- **Bzip2 (`.bz2`)**: Slower, better compression
- **Zip (`.zip`)**: Fast, good for multiple files
- **Tar archives**: Fast, preserves directory structure

### Temporary Files

Archives are extracted to temporary directories:
- Location: System temp directory (e.g., `/tmp/rbn_extract_*`)
- Cleanup: Automatic after processing
- Space needed: Same as uncompressed size

### Memory Usage

The tool still uses chunked reading (100k rows) even for extracted files, so memory usage remains efficient regardless of archive size.

## Troubleshooting

### "Error extracting archive"

**Cause**: Corrupted or unsupported archive format

**Solution**: 
- Verify the archive isn't corrupted: `gzip -t file.gz` or `unzip -t file.zip`
- Check file extension matches actual format
- Try extracting manually to verify contents

### "No files found after extraction"

**Cause**: Archive doesn't contain `.txt` or `.csv` files

**Solution**:
- Check archive contents: `tar -tzf file.tar.gz` or `unzip -l file.zip`
- Ensure files have `.txt` or `.csv` extensions
- RBN logs should be text files with whitespace-separated columns

### Disk Space Issues

**Cause**: Not enough space in temp directory

**Solution**:
- Check available space: `df -h /tmp`
- Free up space or set `TMPDIR` environment variable:
  ```bash
  export TMPDIR=/path/to/larger/disk
  rbn-train --input data/ --output-dir results/
  ```

### Slow Extraction

**Cause**: Large archives or slow disk

**Solution**:
- Use `.gz` instead of `.bz2` for faster extraction
- Extract manually to SSD if available
- Process files individually instead of large archives

## Best Practices

### 1. Keep Archives Organized

```
data/
├── 2023/
│   ├── cqww_wpx.txt.gz
│   └── cqww_dx.txt.gz
├── 2024/
│   ├── cqww_wpx.txt.gz
│   └── cqww_dx.txt.gz
└── 2025/
    └── cqww_wpx.txt.gz
```

Process by year:
```bash
rbn-train --input data/2024/ --output-dir results/2024/
```

### 2. Use Appropriate Compression

- **For storage**: Use `.bz2` (better compression)
- **For processing**: Use `.gz` (faster extraction)
- **For multiple files**: Use `.tar.gz` or `.zip`

### 3. Verify Before Processing

```bash
# Check what's in the archive
tar -tzf rbn_logs.tar.gz | head

# Verify it's not corrupted
gzip -t file.gz

# Then process
rbn-train --input rbn_logs.tar.gz --output-dir results/
```

### 4. Monitor Disk Space

```bash
# Check space before processing large archives
df -h /tmp

# Process with monitoring
rbn-train --input large_archive.tar.gz --output-dir results/
```

## Technical Details

### Extraction Process

1. **Detection**: File extension checked against supported formats
2. **Temporary Directory**: Created in system temp location
3. **Extraction**: Archive extracted using appropriate library
4. **Discovery**: All `.txt` and `.csv` files found recursively
5. **Processing**: Files processed with normal pipeline
6. **Cleanup**: Temporary directory removed automatically

### Libraries Used

- `gzip`: Standard library (gzip files)
- `bz2`: Standard library (bzip2 files)
- `zipfile`: Standard library (zip files)
- `tarfile`: Standard library (tar archives)

No additional dependencies required!

## Summary

✅ **No manual extraction needed** - Just point the tool at your archives  
✅ **Multiple formats supported** - .gz, .bz2, .zip, .tar, .tar.gz, .tar.bz2  
✅ **Automatic cleanup** - Temporary files removed after processing  
✅ **Memory efficient** - Chunked reading even for extracted files  
✅ **Fast processing** - Optimized extraction and parsing  

Simply run:
```bash
rbn-train --input your_compressed_logs/ --output-dir results/
```

And let the tool handle the rest!
