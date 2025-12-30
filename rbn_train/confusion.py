"""Confusion model generation and export."""

import logging
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np

from rbn_train.config import AnalysisConfig
from rbn_train.types import FinalCluster

logger = logging.getLogger(__name__)


@dataclass
class Alphabet:
    """Character set definition with indexing.
    
    Attributes:
        chars: List of characters in the alphabet
        index_by_char: Mapping from character to index
        unknown_char: Character used for unknown/invalid characters
        unknown_index: Index of the unknown character
    """
    chars: List[str]
    index_by_char: Dict[str, int]
    unknown_char: str
    unknown_index: int


class EditOpKind(Enum):
    """Type of edit operation in Levenshtein alignment."""
    MATCH = "match"
    SUB = "sub"
    DEL = "del"
    INS = "ins"


@dataclass
class EditOp:
    """A single edit operation from Levenshtein alignment.
    
    Attributes:
        kind: Type of operation (match, substitution, deletion, insertion)
        true_char: Character from true callsign (None for insertion)
        obs_char: Character from observed callsign (None for deletion)
    """
    kind: EditOpKind
    true_char: Optional[str]
    obs_char: Optional[str]


@dataclass
class ConfusionCounts:
    """Multi-dimensional arrays storing character-level error counts.
    
    Attributes:
        sub_counts: Substitution counts [mode][band][true_char][obs_char]
        del_counts: Deletion counts [mode][band][true_char]
        ins_counts: Insertion counts [mode][band][obs_char]
    """
    sub_counts: np.ndarray
    del_counts: np.ndarray
    ins_counts: np.ndarray


def build_alphabet(charset: str, unknown_char: str) -> Alphabet:
    """Create alphabet with character indexing.
    
    Args:
        charset: String of valid characters
        unknown_char: Character to use for unknown/invalid characters
        
    Returns:
        Alphabet object with character indexing
    """
    chars = list(charset)
    
    # Add unknown_char if not already in charset
    if unknown_char not in chars:
        chars.append(unknown_char)
    
    # Build index mapping
    index_by_char = {ch: idx for idx, ch in enumerate(chars)}
    unknown_index = index_by_char[unknown_char]
    
    return Alphabet(
        chars=chars,
        index_by_char=index_by_char,
        unknown_char=unknown_char,
        unknown_index=unknown_index,
    )


def snr_band_index(snr: float, edges: List[float]) -> int:
    """Find SNR band index for given SNR value.
    
    Args:
        snr: SNR value in dB
        edges: List of SNR band edges in ascending order
        
    Returns:
        Band index i such that edges[i] < snr <= edges[i+1]
    """
    # Find the band where edges[i] < snr <= edges[i+1]
    for i in range(len(edges) - 1):
        if edges[i] < snr <= edges[i + 1]:
            return i
    
    # Fallback: return last band
    return len(edges) - 2


def normalize_call(call: str, alphabet: Alphabet) -> str:
    """Replace unknown characters with unknown_char.
    
    Args:
        call: Callsign to normalize
        alphabet: Alphabet definition
        
    Returns:
        Normalized callsign with unknown characters replaced
    """
    normalized = []
    for char in call:
        if char in alphabet.index_by_char:
            normalized.append(char)
        else:
            normalized.append(alphabet.unknown_char)
    return ''.join(normalized)


def align_calls(true_call: str, obs_call: str) -> List[EditOp]:
    """Perform Levenshtein alignment and extract edit operations.
    
    Uses dynamic programming to find minimum edit distance, then backtraces
    to extract the sequence of edit operations.
    
    Args:
        true_call: True callsign
        obs_call: Observed callsign
        
    Returns:
        List of EditOp objects in left-to-right order
    """
    m, n = len(true_call), len(obs_call)
    
    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if true_call[i - 1] == obs_call[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # Match
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1],  # Substitute
                )
    
    # Backtrace to extract operations
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i == 0:
            # Only insertions left
            operations.append(EditOp(EditOpKind.INS, None, obs_call[j - 1]))
            j -= 1
        elif j == 0:
            # Only deletions left
            operations.append(EditOp(EditOpKind.DEL, true_call[i - 1], None))
            i -= 1
        elif true_call[i - 1] == obs_call[j - 1]:
            # Match
            operations.append(EditOp(EditOpKind.MATCH, true_call[i - 1], obs_call[j - 1]))
            i -= 1
            j -= 1
        else:
            # Find which operation was used
            del_cost = dp[i - 1][j]
            ins_cost = dp[i][j - 1]
            sub_cost = dp[i - 1][j - 1]
            
            min_cost = min(del_cost, ins_cost, sub_cost)
            
            if sub_cost == min_cost:
                # Substitution
                operations.append(EditOp(EditOpKind.SUB, true_call[i - 1], obs_call[j - 1]))
                i -= 1
                j -= 1
            elif del_cost == min_cost:
                # Deletion
                operations.append(EditOp(EditOpKind.DEL, true_call[i - 1], None))
                i -= 1
            else:
                # Insertion
                operations.append(EditOp(EditOpKind.INS, None, obs_call[j - 1]))
                j -= 1
    
    # Reverse to get left-to-right order
    operations.reverse()
    
    return operations


def build_confusion_and_priors(final_clusters: List[FinalCluster],
                               config: AnalysisConfig) -> Tuple[ConfusionCounts, 'Priors', Alphabet]:
    """Generate confusion matrices and priors from final clusters.
    
    Args:
        final_clusters: List of FinalCluster objects
        config: Analysis configuration
        
    Returns:
        Tuple of (ConfusionCounts, Priors, Alphabet)
    """
    from rbn_train.priors import Priors
    
    # Build alphabet
    alphabet = build_alphabet(config.charset, config.unknown_char)
    
    # Build mode index
    mode_index = {mode: idx for idx, mode in enumerate(config.modes)}
    
    # Initialize counts
    num_modes = len(config.modes)
    num_bands = len(config.snr_bands) - 1
    num_chars = len(alphabet.chars)
    
    sub_counts = np.zeros((num_modes, num_bands, num_chars, num_chars), dtype=np.int64)
    del_counts = np.zeros((num_modes, num_bands, num_chars), dtype=np.int64)
    ins_counts = np.zeros((num_modes, num_bands, num_chars), dtype=np.int64)
    
    # Initialize priors
    priors = Priors(counts={})
    
    # Process each final cluster
    for fc in final_clusters:
        if fc.true_call_final is None:
            continue
        
        # Normalize true call
        true_call_norm = normalize_call(fc.true_call_final, alphabet)
        
        # Update priors
        priors.counts[true_call_norm] = priors.counts.get(true_call_norm, 0) + 1
        
        # Process each spot in cluster
        for spot in fc.cluster.spots:
            # Get mode index
            if spot.mode not in mode_index:
                continue
            m_idx = mode_index[spot.mode]
            
            # Get SNR band index
            b_idx = snr_band_index(spot.snr_db, config.snr_bands)
            
            # Normalize observed call
            obs_call_norm = normalize_call(spot.dx_call, alphabet)
            
            # Align calls
            ops = align_calls(true_call_norm, obs_call_norm)
            
            # Process operations
            for op in ops:
                if op.kind == EditOpKind.MATCH:
                    continue  # Skip matches
                
                elif op.kind == EditOpKind.SUB:
                    if op.true_char is not None and op.obs_char is not None:
                        ti = alphabet.index_by_char[op.true_char]
                        oi = alphabet.index_by_char[op.obs_char]
                        sub_counts[m_idx, b_idx, ti, oi] += 1
                
                elif op.kind == EditOpKind.DEL:
                    if op.true_char is not None:
                        ti = alphabet.index_by_char[op.true_char]
                        del_counts[m_idx, b_idx, ti] += 1
                
                elif op.kind == EditOpKind.INS:
                    if op.obs_char is not None:
                        oi = alphabet.index_by_char[op.obs_char]
                        ins_counts[m_idx, b_idx, oi] += 1
    
    confusion = ConfusionCounts(
        sub_counts=sub_counts,
        del_counts=del_counts,
        ins_counts=ins_counts,
    )
    
    return confusion, priors, alphabet


def export_confusion_model(confusion: ConfusionCounts,
                          alphabet: Alphabet,
                          modes: List[str],
                          snr_bands: List[float],
                          output_path: Path) -> None:
    """Write confusion_model.json.
    
    Args:
        confusion: ConfusionCounts object
        alphabet: Alphabet definition
        modes: List of mode names
        snr_bands: List of SNR band edges
        output_path: Path to write JSON file
    """
    data = {
        "modes": modes,
        "snr_band_edges": snr_bands,
        "alphabet": ''.join(alphabet.chars),
        "unknown_char": alphabet.unknown_char,
        "sub_counts": confusion.sub_counts.tolist(),
        "del_counts": confusion.del_counts.tolist(),
        "ins_counts": confusion.ins_counts.tolist(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# Worker function for parallel processing (must be at module level for pickling)
def _build_confusion_batch(args: Tuple[List[FinalCluster], dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Process a batch of final clusters for confusion model building.

    Args:
        args: Tuple of (final_clusters, params_dict) where params_dict contains
              alphabet_chars, unknown_char, mode_index, snr_bands

    Returns:
        Tuple of (sub_counts, del_counts, ins_counts, prior_counts)
    """
    final_clusters, params = args

    # Reconstruct alphabet
    chars = list(params['alphabet_chars'])
    index_by_char = {ch: idx for idx, ch in enumerate(chars)}
    unknown_char = params['unknown_char']
    alphabet = Alphabet(
        chars=chars,
        index_by_char=index_by_char,
        unknown_char=unknown_char,
        unknown_index=index_by_char[unknown_char],
    )

    mode_index = params['mode_index']
    snr_bands = params['snr_bands']

    # Initialize partial counts
    num_modes = params['num_modes']
    num_bands = params['num_bands']
    num_chars = len(chars)

    sub_counts = np.zeros((num_modes, num_bands, num_chars, num_chars), dtype=np.int64)
    del_counts = np.zeros((num_modes, num_bands, num_chars), dtype=np.int64)
    ins_counts = np.zeros((num_modes, num_bands, num_chars), dtype=np.int64)
    prior_counts: Dict[str, int] = {}

    # Process each final cluster
    for fc in final_clusters:
        if fc.true_call_final is None:
            continue

        # Normalize true call
        true_call_norm = normalize_call(fc.true_call_final, alphabet)

        # Update priors
        prior_counts[true_call_norm] = prior_counts.get(true_call_norm, 0) + 1

        # Process each spot in cluster
        for spot in fc.cluster.spots:
            # Get mode index
            if spot.mode not in mode_index:
                continue
            m_idx = mode_index[spot.mode]

            # Get SNR band index
            b_idx = snr_band_index(spot.snr_db, snr_bands)

            # Normalize observed call
            obs_call_norm = normalize_call(spot.dx_call, alphabet)

            # Align calls
            ops = align_calls(true_call_norm, obs_call_norm)

            # Process operations
            for op in ops:
                if op.kind == EditOpKind.MATCH:
                    continue  # Skip matches

                elif op.kind == EditOpKind.SUB:
                    if op.true_char is not None and op.obs_char is not None:
                        ti = alphabet.index_by_char[op.true_char]
                        oi = alphabet.index_by_char[op.obs_char]
                        sub_counts[m_idx, b_idx, ti, oi] += 1

                elif op.kind == EditOpKind.DEL:
                    if op.true_char is not None:
                        ti = alphabet.index_by_char[op.true_char]
                        del_counts[m_idx, b_idx, ti] += 1

                elif op.kind == EditOpKind.INS:
                    if op.obs_char is not None:
                        oi = alphabet.index_by_char[op.obs_char]
                        ins_counts[m_idx, b_idx, oi] += 1

    return sub_counts, del_counts, ins_counts, prior_counts


def build_confusion_and_priors_parallel(
    final_clusters: List[FinalCluster],
    config: AnalysisConfig,
    workers: int = 1,
    chunk_size: int = 1000
) -> Tuple[ConfusionCounts, 'Priors', Alphabet]:
    """Generate confusion matrices and priors (parallel version).

    Uses multiprocessing to distribute cluster processing across workers,
    then merges partial counts.

    Args:
        final_clusters: List of FinalCluster objects
        config: Analysis configuration
        workers: Number of worker processes
        chunk_size: Number of clusters per worker batch

    Returns:
        Tuple of (ConfusionCounts, Priors, Alphabet)
    """
    from rbn_train.priors import Priors

    # Sequential processing for single worker or empty input
    if workers <= 1 or not final_clusters:
        return build_confusion_and_priors(final_clusters, config)

    logger.info(f"Building confusion model for {len(final_clusters)} clusters with {workers} workers")

    from concurrent.futures import ProcessPoolExecutor

    # Build alphabet
    alphabet = build_alphabet(config.charset, config.unknown_char)

    # Build mode index
    mode_index = {mode: idx for idx, mode in enumerate(config.modes)}

    # Prepare parameters for workers
    params = {
        'alphabet_chars': ''.join(alphabet.chars),
        'unknown_char': alphabet.unknown_char,
        'mode_index': mode_index,
        'snr_bands': config.snr_bands,
        'num_modes': len(config.modes),
        'num_bands': len(config.snr_bands) - 1,
    }

    # Split final_clusters into batches
    batches = []
    for i in range(0, len(final_clusters), chunk_size):
        batch = final_clusters[i:i + chunk_size]
        batches.append((batch, params))

    logger.debug(f"Split into {len(batches)} batches of up to {chunk_size} clusters each")

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_build_confusion_batch, batches))

    # Merge results
    num_modes = len(config.modes)
    num_bands = len(config.snr_bands) - 1
    num_chars = len(alphabet.chars)

    total_sub = np.zeros((num_modes, num_bands, num_chars, num_chars), dtype=np.int64)
    total_del = np.zeros((num_modes, num_bands, num_chars), dtype=np.int64)
    total_ins = np.zeros((num_modes, num_bands, num_chars), dtype=np.int64)
    total_priors: Dict[str, int] = {}

    for sub_counts, del_counts, ins_counts, prior_counts in results:
        total_sub += sub_counts
        total_del += del_counts
        total_ins += ins_counts
        for call, count in prior_counts.items():
            total_priors[call] = total_priors.get(call, 0) + count

    confusion = ConfusionCounts(
        sub_counts=total_sub,
        del_counts=total_del,
        ins_counts=total_ins,
    )

    priors = Priors(counts=total_priors)

    logger.info(f"Completed parallel confusion model building")
    return confusion, priors, alphabet
