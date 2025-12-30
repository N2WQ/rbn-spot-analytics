"""Global priors generation and export."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import json


@dataclass
class Priors:
    """Global callsign frequency counts.
    
    Attributes:
        counts: Dictionary mapping normalized callsigns to their occurrence counts
    """
    counts: Dict[str, int] = field(default_factory=dict)


def export_priors(priors: Priors, output_path: Path) -> None:
    """Write priors.json.
    
    Args:
        priors: Priors object with callsign counts
        output_path: Path to write JSON file
    """
    data = {
        "calls": priors.counts
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
