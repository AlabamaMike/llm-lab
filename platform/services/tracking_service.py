"""W&B tracking service for experiment metrics."""
import os
from typing import Any, Optional

import wandb


class TrackingService:
    """Service for tracking experiments with Weights & Biases."""

    def __init__(self, api_key: str, project: str, entity: Optional[str] = None):
        """
        Initialize tracking service.

        Args:
            api_key: W&B API key
            project: W&B project name
            entity: Optional W&B entity (username or team)
        """
        self.api_key = api_key
        self.project = project
        self.entity = entity
        self._current_run = None

        # Set API key in environment
        os.environ["WANDB_API_KEY"] = api_key

    def create_run(
        self,
        name: str,
        config: dict,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Create a new W&B run.

        Args:
            name: Run name
            config: Configuration dict (model params, etc.)
            tags: Optional list of tags
            notes: Optional notes

        Returns:
            W&B run ID
        """
        self._current_run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config,
            tags=tags or [],
            notes=notes,
        )

        return self._current_run.id

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to current run.

        Args:
            metrics: Dict of metric name -> value
            step: Optional step number for time series
        """
        if not self._current_run:
            raise RuntimeError("No active run. Call create_run() first.")

        self._current_run.log(metrics, step=step)

    def log_artifact(
        self, artifact_path: str, artifact_type: str, name: str
    ) -> None:
        """
        Log an artifact (file, dataset, model) to W&B.

        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (dataset, model, etc.)
            name: Artifact name
        """
        if not self._current_run:
            raise RuntimeError("No active run. Call create_run() first.")

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(artifact_path)
        self._current_run.log_artifact(artifact)

    def finish_run(self, exit_code: int = 0) -> None:
        """
        Finish the current run.

        Args:
            exit_code: Exit code (0 for success, non-zero for failure)
        """
        if self._current_run:
            self._current_run.finish(exit_code=exit_code)
            self._current_run = None

    def update_summary(self, summary: dict[str, Any]) -> None:
        """
        Update run summary (final metrics).

        Args:
            summary: Dict of summary metrics
        """
        if not self._current_run:
            raise RuntimeError("No active run. Call create_run() first.")

        for key, value in summary.items():
            self._current_run.summary[key] = value

    @property
    def run_url(self) -> Optional[str]:
        """Get URL for current run."""
        if self._current_run:
            return self._current_run.url
        return None
