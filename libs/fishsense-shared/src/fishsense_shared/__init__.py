"""Shared helpers for FishSense Lite services."""

from fishsense_shared.auto_accept_timeouts import (
    GATE_ACTIVITY_TIMEOUT,
    GATE_CHILD_EXECUTION_TIMEOUT,
    GATE_EXECUTION_TIMEOUT,
    GATE_QUEUE_WAIT_TIMEOUT,
)
from fishsense_shared.config import (
    IS_DOCKER,
    get_config_path,
    get_log_path,
    path_validator,
    url_condition,
)
from fishsense_shared.exception_group import ExceptionGroupErrorLogging
from fishsense_shared.laser_predictor import (
    LASER_PREDICTOR_VERSION,
    laser_model_version_tag,
)
from fishsense_shared.laser_region import (
    DEFAULT_LASER_BBOX,
    LASER_REGION_POLYGON,
    WORKING_DEPTH_RANGE_M,
    point_in_laser_region,
)
from fishsense_shared.logging import configure_log_handler, configure_logging
from fishsense_shared.ingest_contracts import (
    DuplicateOverlap,
    IngestDiveRequest,
    IngestPreflight,
    IngestProgress,
    IngestReport,
    PreflightImage,
    RejectedImage,
    SubfolderReport,
)
from fishsense_shared.preprocess_contracts import (
    ClusterDiveFrameImage,
    ClusterDiveFramesInput,
    LaserAutoAcceptSummary,
    LaserPredictionResult,
    PredictLaserImage,
    PredictLaserImagesInput,
    PredictSlateImage,
    PredictSlateImagesInput,
    PreprocessHeadtailImagesInput,
    PreprocessLaserImagesInput,
    PreprocessSlateImagesInput,
    PreprocessSpeciesImagesInput,
    SlatePredictionResult,
)
from fishsense_shared.task_queues import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
    DATA_PROCESSING_TASK_QUEUE,
)
from fishsense_shared.temporal import (
    build_tls_config,
    ensure_schedule,
    temporal_namespace,
)

__all__ = [
    "IS_DOCKER",
    "GATE_ACTIVITY_TIMEOUT",
    "GATE_CHILD_EXECUTION_TIMEOUT",
    "GATE_EXECUTION_TIMEOUT",
    "GATE_QUEUE_WAIT_TIMEOUT",
    "DEFAULT_LASER_BBOX",
    "LASER_PREDICTOR_VERSION",
    "laser_model_version_tag",
    "LASER_REGION_POLYGON",
    "WORKING_DEPTH_RANGE_M",
    "point_in_laser_region",
    "DuplicateOverlap",
    "IngestDiveRequest",
    "IngestPreflight",
    "IngestProgress",
    "IngestReport",
    "PreflightImage",
    "RejectedImage",
    "SubfolderReport",
    "DATA_PROCESSING_GPU_TASK_QUEUE",
    "DATA_PROCESSING_TASK_QUEUE",
    "ClusterDiveFrameImage",
    "ClusterDiveFramesInput",
    "ExceptionGroupErrorLogging",
    "LaserAutoAcceptSummary",
    "LaserPredictionResult",
    "PredictLaserImage",
    "PredictLaserImagesInput",
    "PredictSlateImage",
    "PredictSlateImagesInput",
    "PreprocessHeadtailImagesInput",
    "PreprocessLaserImagesInput",
    "PreprocessSlateImagesInput",
    "PreprocessSpeciesImagesInput",
    "SlatePredictionResult",
    "build_tls_config",
    "configure_log_handler",
    "configure_logging",
    "ensure_schedule",
    "get_config_path",
    "get_log_path",
    "path_validator",
    "temporal_namespace",
    "url_condition",
]
