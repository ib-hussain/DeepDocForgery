"""Input branches used by the DeepDocForgery pipeline."""

from .degradationEstimator import (
    DegradationEstimatorOutput,
    DegradationLevelOutput,
    DegradationTargets,
    FrontEndOutput,
    InputForensicsFrontEnd,
    MultiScaleDegradationLoss,
    RegionalDegradationEstimator,
)
from .freqFeatures import (
    DCTBranchOutput,
    DCTFeaturePyramid,
    ExactDCTBatch,
    ExactJPEGDCTReader,
    JPEGMetadata,
    PixelDomainDCT,
)

__all__ = [
    "DCTBranchOutput",
    "DCTFeaturePyramid",
    "DegradationEstimatorOutput",
    "DegradationLevelOutput",
    "DegradationTargets",
    "ExactDCTBatch",
    "ExactJPEGDCTReader",
    "FrontEndOutput",
    "InputForensicsFrontEnd",
    "JPEGMetadata",
    "MultiScaleDegradationLoss",
    "PixelDomainDCT",
    "RegionalDegradationEstimator",
]
