"""Turn dense tamper probabilities into auditable region instances."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from torch import Tensor


@dataclass(frozen=True)
class InstancePrediction:
    """One connected suspicious region in pixel coordinates."""

    image_index: int
    instance_id: int
    score: float
    area: int
    box_xyxy: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _components(binary: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    result: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while queue:
                current_y, current_x = queue.popleft()
                component.append((current_y, current_x))
                for offset_y, offset_x in offsets:
                    next_y, next_x = current_y + offset_y, current_x + offset_x
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and binary[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
            result.append(component)
    return result


def extract_instances(
    probabilities: Tensor,
    *,
    threshold: float = 0.5,
    minimum_area: int = 16,
    connectivity: int = 8,
) -> list[list[InstancePrediction]]:
    """Extract connected components and bounding boxes for every batch item."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if minimum_area < 1:
        raise ValueError("minimum_area must be positive")
    if probabilities.ndim == 3:
        probabilities = probabilities.unsqueeze(1)
    if probabilities.ndim != 4 or probabilities.shape[1] != 1:
        raise ValueError("probabilities must have shape [B,1,H,W] or [B,H,W]")

    values = probabilities.detach().float().cpu().numpy()[:, 0]
    all_instances: list[list[InstancePrediction]] = []
    for image_index, probability in enumerate(values):
        instances: list[InstancePrediction] = []
        for component in _components(probability >= threshold, connectivity):
            if len(component) < minimum_area:
                continue
            ys = np.fromiter((point[0] for point in component), dtype=np.int64)
            xs = np.fromiter((point[1] for point in component), dtype=np.int64)
            score = float(probability[ys, xs].mean())
            instances.append(
                InstancePrediction(
                    image_index=image_index,
                    instance_id=len(instances),
                    score=score,
                    area=len(component),
                    box_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                )
            )
        instances.sort(key=lambda value: (-value.score, -value.area, value.box_xyxy))
        instances = [
            InstancePrediction(
                image_index=value.image_index,
                instance_id=index,
                score=value.score,
                area=value.area,
                box_xyxy=value.box_xyxy,
            )
            for index, value in enumerate(instances)
        ]
        all_instances.append(instances)
    return all_instances
