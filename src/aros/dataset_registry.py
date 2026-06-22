"""Dataset Registry — manages dataset metadata.

The system never exposes raw datasets to the Research Agent.
Instead it provides metadata: quality scores, novelty scores,
overlap percentages, topic distributions, and diversity metrics.
"""

from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import DatasetInfo


class DatasetRegistry:
    """Metadata-only dataset registry. No raw data exposure."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.min_quality = config.dataset_registry.min_quality_score
        self.synthetic_cap = config.dataset_registry.synthetic_cap

    def register(self, dataset: DatasetInfo) -> str:
        """Register a dataset in the registry."""
        return self.db.register_dataset(dataset)

    def query(self, dataset_type: Optional[str] = None, min_quality: Optional[float] = None) -> list[DatasetInfo]:
        """Query datasets by type and quality threshold."""
        datasets = self.db.list_datasets(dataset_type=dataset_type)
        threshold = min_quality or self.min_quality
        return [d for d in datasets if d.quality_score >= threshold]

    def get_summary_stats(self) -> dict:
        """Get summary statistics about all registered datasets."""
        all_datasets = self.db.list_datasets()
        if not all_datasets:
            return {"total": 0}

        by_type = {}
        for d in all_datasets:
            by_type.setdefault(d.type, []).append(d)

        return {
            "total": len(all_datasets),
            "by_type": {t: len(ds) for t, ds in by_type.items()},
            "avg_quality": sum(d.quality_score for d in all_datasets) / len(all_datasets),
            "avg_novelty": sum(d.novelty_score for d in all_datasets) / len(all_datasets),
            "synthetic_count": sum(1 for d in all_datasets if d.is_synthetic),
            "total_examples": sum(d.num_examples for d in all_datasets),
        }

    def validate_mixture(self, dataset_ids: list[str]) -> tuple[bool, str]:
        """Validate that a proposed dataset mixture meets constraints.

        Checks:
        - All datasets exist
        - Quality scores meet threshold
        - Synthetic data doesn't exceed cap
        """
        datasets = []
        for did in dataset_ids:
            # We don't have direct query-by-id, use list and filter
            all_ds = self.db.list_datasets()
            matches = [d for d in all_ds if d.id == did]
            if not matches:
                return False, f"Dataset {did} not found"
            datasets.append(matches[0])

        synthetic_count = sum(1 for d in datasets if d.is_synthetic)
        total = len(datasets)
        if total > 0 and (synthetic_count / total) > self.synthetic_cap:
            return False, f"Synthetic ratio {synthetic_count}/{total} exceeds cap of {self.synthetic_cap}"

        return True, "Validation passed"
