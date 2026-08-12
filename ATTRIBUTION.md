# Attribution and reuse boundary

This milestone is licensed under GPL-3.0-or-later. It is a new integration for
the DeepDocForgery architecture, not a fork of one upstream repository.

| Upstream work | Licence observed | Use in this milestone |
| --- | --- | --- |
| [ADCD-Net](https://github.com/KahimWong/ADCD-Net) | MIT | The quantization-table extraction helper is adapted and extended to three YCbCr components. The learned quantization conditioning and multi-scale frequency pyramid are architectural adaptations. The upstream MIT notice is preserved in `third_party/ADCD-Net-LICENSE`. |
| [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | Apache-2.0 | Architectural reference for decoupled JPEG-quality prediction and double-JPEG supervision. This milestone does not copy its restoration decoder or source text. |
| [DocTamper](https://github.com/qcf-568/DocTamper) | No clear repository-wide software licence found during the 2026-08-12 review | Paper/method reference only. No source copied. Its Frequency Perception Head motivated frequency-aware document analysis. |
| [DanceText](https://github.com/qcf-568/DanceText) | No licence relied upon in this milestone | Future decoder/baseline reference only; no code copied into these two input branches. |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause | Planned reference for later compound-degradation data generation; no code integrated yet. |
| [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) | CC-BY-4.0 shown by the repository | Future benchmark adapter target only; no code integrated yet. |

## Papers to cite

- Wong et al., “ADCD-Net: Robust Document Image Forgery Localization via
  Adaptive DCT Feature and Hierarchical Content Disentanglement,” ICCV 2025.
- Jiang, Zhang, and Timofte, “Towards Flexible Blind JPEG Artifacts Removal,”
  ICCV 2021.
- Qu et al., “Towards Robust Tampered Text Detection in Document Image: New
  Dataset and New Solution,” CVPR 2023.

Before publishing a paper, add the citations for DanceText and any later-used
components to the manuscript bibliography. Before copying more upstream source,
re-check the exact commit and licence rather than relying on this snapshot.

