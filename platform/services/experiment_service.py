"""Business logic for experiment management."""
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from platform.models.experiment import Experiment, ExperimentStatus


class ExperimentService:
    """Service for managing LLM experiments."""

    def __init__(self, db: Session):
        """
        Initialize experiment service.

        Args:
            db: Database session
        """
        self.db = db

    def create_experiment(
        self,
        user_id: int,
        name: str,
        provider: str,
        model_name: str,
        model_params: dict,
        description: Optional[str] = None,
        prompt_repo_id: Optional[int] = None,
        prompt_file_path: Optional[str] = None,
        dataset_gcs_path: Optional[str] = None,
        dataset_inline: Optional[dict] = None,
        evaluation_config: Optional[dict] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            user_id: ID of user creating the experiment
            name: Experiment name
            provider: LLM provider (openai, anthropic)
            model_name: Model name (gpt-4, etc.)
            model_params: Model configuration (temperature, max_tokens, etc.)
            description: Optional description
            prompt_repo_id: Optional prompt repository ID
            prompt_file_path: Optional path to prompt file
            dataset_gcs_path: Optional GCS path to dataset
            dataset_inline: Optional inline dataset
            evaluation_config: Optional evaluation configuration

        Returns:
            Created experiment
        """
        experiment = Experiment(
            user_id=user_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            provider=provider,
            model_name=model_name,
            model_params=model_params,
            prompt_repo_id=prompt_repo_id,
            prompt_file_path=prompt_file_path,
            dataset_gcs_path=dataset_gcs_path,
            dataset_inline=dataset_inline,
            evaluation_config=evaluation_config,
        )

        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)

        return experiment

    def get_experiment(self, experiment_id: int, user_id: int) -> Optional[Experiment]:
        """
        Get an experiment by ID (with user access check).

        Args:
            experiment_id: Experiment ID
            user_id: User ID for access check

        Returns:
            Experiment or None if not found/not authorized
        """
        return (
            self.db.query(Experiment)
            .filter(Experiment.id == experiment_id, Experiment.user_id == user_id)
            .first()
        )

    def list_experiments(
        self,
        user_id: int,
        status: Optional[ExperimentStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Experiment]:
        """
        List experiments for a user.

        Args:
            user_id: User ID
            status: Optional status filter
            limit: Max number of results
            offset: Pagination offset

        Returns:
            List of experiments
        """
        query = self.db.query(Experiment).filter(Experiment.user_id == user_id)

        if status:
            query = query.filter(Experiment.status == status)

        return query.order_by(Experiment.created_at.desc()).limit(limit).offset(offset).all()

    def submit_experiment(self, experiment: Experiment) -> None:
        """
        Submit an experiment for execution.

        Args:
            experiment: Experiment to submit

        Raises:
            ValueError: If experiment is not in DRAFT status
        """
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot submit experiment with status {experiment.status}")

        experiment.status = ExperimentStatus.QUEUED
        self.db.commit()

    def update_experiment_status(
        self,
        experiment: Experiment,
        status: ExperimentStatus,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """
        Update experiment status and timestamps.

        Args:
            experiment: Experiment to update
            status: New status
            started_at: Optional start time
            completed_at: Optional completion time
        """
        experiment.status = status

        if started_at:
            experiment.started_at = started_at

        if completed_at:
            experiment.completed_at = completed_at

        self.db.commit()

    def update_experiment_results(
        self,
        experiment: Experiment,
        total_cost_usd: float,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> None:
        """
        Update experiment with execution results.

        Args:
            experiment: Experiment to update
            total_cost_usd: Total cost
            total_input_tokens: Total input tokens
            total_output_tokens: Total output tokens
        """
        experiment.total_cost_usd = total_cost_usd
        experiment.total_input_tokens = total_input_tokens
        experiment.total_output_tokens = total_output_tokens

        self.db.commit()
