"""Input branches used by the DeepDocForgery pipeline."""

from .dataModelling import ShuffledPatchBatch, shuffle_artifact_patches
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
from .spatialFeature import (
    ADNSupervision,
    ArtifactDecoupleNetwork,
    ArtifactDecouplingLoss,
    HierarchicalViTBackbone,
    SpatialBranchOutput,
    SpatialFeaturePyramid,
)

__all__ = [
    "DCTBranchOutput",
    "DCTFeaturePyramid",
    "ADNSupervision",
    "ArtifactDecoupleNetwork",
    "ArtifactDecouplingLoss",
    "DegradationEstimatorOutput",
    "DegradationLevelOutput",
    "DegradationTargets",
    "ExactDCTBatch",
    "ExactJPEGDCTReader",
    "FrontEndOutput",
    "HierarchicalViTBackbone",
    "InputForensicsFrontEnd",
    "JPEGMetadata",
    "MultiScaleDegradationLoss",
    "PixelDomainDCT",
    "RegionalDegradationEstimator",
    "ShuffledPatchBatch",
    "SpatialBranchOutput",
    "SpatialFeaturePyramid",
    "shuffle_artifact_patches",
]
