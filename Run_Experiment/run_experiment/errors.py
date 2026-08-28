"""Operator-facing synchronized acquisition failures."""


class ExperimentError(Exception):
    """Base error for expected configuration/acquisition failures."""


class ExperimentConfigurationError(ExperimentError):
    """Run_Experiment configuration or operator metadata is invalid."""


class BusyTriggerError(ExperimentError):
    """A trigger arrived while the orchestrator was not ARMED."""


class SynchronizedAcquisitionError(ExperimentError):
    """A prepared camera could not provide its requested evidence."""
