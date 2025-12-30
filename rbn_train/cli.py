"""Command-line interface for RBN analysis."""

import argparse
import logging
from pathlib import Path
import sys

from rbn_train.config import load_config, finalize_config, validate_config, AnalysisConfig
from rbn_train.io import resolve_input_paths, load_spots
from rbn_train.clustering import build_micro_clusters, label_provisional_truth, label_provisional_truth_parallel
from rbn_train.stability import build_stability_bins, compute_stable_calls, refine_truth, refine_truth_parallel
from rbn_train.confusion import build_confusion_and_priors, build_confusion_and_priors_parallel, export_confusion_model
from rbn_train.priors import export_priors
from rbn_train.reliability import (
    compute_spotter_reliability,
    compute_spotter_reliability_parallel,
    export_spotter_reliability,
    export_spotter_reliability_simple,
    export_spotter_reliability_by_mode
)
from rbn_train.band_priors import (
    compute_band_priors,
    compute_segment_priors,
    export_band_priors,
    export_segment_priors,
    export_band_priors_simple
)
from rbn_train.parallel import get_worker_count


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='RBN Callsign Correction Analytics Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  rbn-train --input data/cqww_2023.csv --output-dir output/
  
  # Directory
  rbn-train --input data/ --output-dir output/
  
  # Glob pattern
  rbn-train --input "data/cqww_*.csv" --output-dir output/
  
  # With config file
  rbn-train --input data/ --output-dir output/ --config config.json
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file, directory, or glob pattern'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for confusion_model.json and priors.json'
    )
    
    parser.add_argument(
        '--config',
        help='Path to JSON configuration file (optional)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes for parallel processing (default: 1, 0=auto-detect)'
    )

    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full analysis pipeline.
    
    Args:
        args: Parsed command-line arguments
    """
    try:
        # Load and finalize configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        config = finalize_config(config)
        validate_config(config)
        logger.info(f"Configuration loaded: {len(config.modes)} modes, {len(config.snr_bands)-1} SNR bands")

        # Resolve worker count
        workers = get_worker_count(args.workers) if args.workers == 0 else args.workers
        chunk_size = config.parallel_chunk_size
        if workers > 1:
            logger.info(f"Parallel processing enabled: {workers} workers, chunk size {chunk_size}")
        
        # Resolve input paths
        logger.info(f"Resolving input: {args.input}")
        paths = resolve_input_paths(args.input)
        logger.info(f"Found {len(paths)} CSV files to process")
        
        # Load spots
        logger.info("Loading spots from CSV files...")
        spots = load_spots(paths, config)
        logger.info(f"Loaded {len(spots)} spots")
        
        if len(spots) == 0:
            logger.warning("No spots loaded. Exiting.")
            return
        
        # Pass A: Build micro-clusters
        logger.info("Building micro-clusters...")
        clusters = build_micro_clusters(spots, config)
        logger.info(f"Created {len(clusters)} micro-clusters")
        
        # Pass A: Label provisional truth
        logger.info("Determining provisional truth...")
        labeled_clusters = label_provisional_truth_parallel(
            clusters, config, workers=workers, chunk_size=chunk_size
        )
        provisional_count = sum(1 for lc in labeled_clusters if lc.provisional_true_call is not None)
        logger.info(f"Assigned provisional truth to {provisional_count}/{len(labeled_clusters)} clusters")
        
        # Pass B: Build stability bins
        logger.info("Building stability bins...")
        stability_bins = build_stability_bins(labeled_clusters, config)
        logger.info(f"Created {len(stability_bins)} stability bins")
        
        # Pass B: Compute stable calls
        logger.info("Computing stable calls...")
        stable_bins = compute_stable_calls(stability_bins, config)
        stable_count = sum(1 for sb in stable_bins.values() if sb.stable_call is not None)
        logger.info(f"Determined stable calls for {stable_count}/{len(stable_bins)} bins")
        
        # Pass B: Refine truth
        logger.info("Refining final truth...")
        final_clusters = refine_truth_parallel(
            labeled_clusters, stable_bins, config, workers=workers, chunk_size=chunk_size
        )
        final_count = sum(1 for fc in final_clusters if fc.true_call_final is not None)
        logger.info(f"Assigned final truth to {final_count}/{len(final_clusters)} clusters")
        
        # Pass C: Build confusion model and priors
        logger.info("Building confusion model and priors...")
        confusion, priors, alphabet = build_confusion_and_priors_parallel(
            final_clusters, config, workers=workers, chunk_size=chunk_size
        )
        logger.info(f"Generated confusion model with {len(alphabet.chars)} characters")
        logger.info(f"Generated priors for {len(priors.counts)} callsigns")

        # Ensure output directory exists
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Export confusion model
        confusion_path = output_dir / "confusion_model.json"
        logger.info(f"Writing confusion model to {confusion_path}")
        export_confusion_model(confusion, alphabet, config.modes, config.snr_bands, confusion_path)

        # Export priors
        priors_path = output_dir / "priors.json"
        logger.info(f"Writing priors to {priors_path}")
        export_priors(priors, priors_path)

        # Track output files for summary
        output_files = [confusion_path, priors_path]

        # Extended Analysis: Spotter Reliability
        if config.output_spotter_reliability:
            logger.info("Computing spotter reliability...")
            reliability = compute_spotter_reliability_parallel(
                final_clusters,
                min_spots=config.min_spotter_spots,
                compute_by_band=True,
                compute_by_mode=True,
                workers=workers,
                chunk_size=chunk_size
            )

            # Export JSON format
            reliability_path = output_dir / "spotter_reliability.json"
            logger.info(f"Writing spotter reliability to {reliability_path}")
            exported, filtered = export_spotter_reliability(
                reliability, reliability_path,
                include_by_band=True,
                include_by_mode=True
            )
            output_files.append(reliability_path)

            # Export simple text format for Go integration
            if config.output_simple_formats:
                reliability_simple_path = output_dir / "spotter_reliability.txt"
                logger.info(f"Writing spotter reliability (simple) to {reliability_simple_path}")
                export_spotter_reliability_simple(reliability, reliability_simple_path)
                output_files.append(reliability_simple_path)

                # Export mode-specific text files
                logger.info("Writing mode-specific spotter reliability files...")
                mode_counts = export_spotter_reliability_by_mode(reliability, output_dir)
                for mode in mode_counts:
                    output_files.append(output_dir / f"spotter_reliability_{mode.lower()}.txt")

        # Extended Analysis: Band Priors
        if config.output_band_priors:
            logger.info("Computing band-specific priors...")
            band_priors = compute_band_priors(final_clusters, alphabet)

            # Export JSON format
            band_priors_path = output_dir / "band_priors.json"
            logger.info(f"Writing band priors to {band_priors_path}")
            export_band_priors(band_priors, band_priors_path)
            output_files.append(band_priors_path)

            # Export simple text format for Go integration
            if config.output_simple_formats:
                band_priors_simple_path = output_dir / "call_quality_priors.txt"
                logger.info(f"Writing band priors (simple) to {band_priors_simple_path}")
                export_band_priors_simple(band_priors, band_priors_simple_path)
                output_files.append(band_priors_simple_path)

        # Extended Analysis: Segment Priors
        if config.output_segment_priors:
            logger.info("Computing frequency segment priors...")
            segment_priors = compute_segment_priors(final_clusters, alphabet=alphabet)

            # Export JSON format
            segment_priors_path = output_dir / "segment_priors.json"
            logger.info(f"Writing segment priors to {segment_priors_path}")
            export_segment_priors(segment_priors, segment_priors_path)
            output_files.append(segment_priors_path)

        # Summary statistics
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total spots processed: {len(spots)}")
        logger.info(f"Micro-clusters created: {len(clusters)}")
        logger.info(f"Clusters with provisional truth: {provisional_count}")
        logger.info(f"Clusters with final truth: {final_count}")
        logger.info(f"Unique callsigns in priors: {len(priors.counts)}")
        if config.output_spotter_reliability:
            logger.info(f"Unique skimmers analyzed: {len(reliability.global_stats)}")
        if config.output_band_priors:
            logger.info(f"Bands with priors: {len(band_priors.bands_seen)}")
        if config.output_segment_priors:
            logger.info(f"Frequency segments: {len(segment_priors.by_segment)}")
        logger.info(f"Output files:")
        for f in output_files:
            logger.info(f"  - {f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Entry point for rbn-train command."""
    args = parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
