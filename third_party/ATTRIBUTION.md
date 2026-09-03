# Attribution and reuse boundary

This project is licensed under GPL-3.0-or-later. It is a new integration for
the DeepDocForgery architecture, not a fork of one upstream repository.

| Upstream work | Licence observed | Use in this milestone |
| --- | --- | --- |
| [ADCD-Net](https://github.com/KahimWong/ADCD-Net) | MIT | The quantization-table extraction helper is adapted and extended to three YCbCr components. The learned quantization conditioning and multi-scale frequency pyramid are architectural adaptations. The upstream MIT notice is preserved in `third_party/ADCD-Net-LICENSE`. |
| [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | Apache-2.0 | Architectural reference for decoupled JPEG-quality prediction and double-JPEG supervision. This milestone does not copy its restoration decoder or source text. |
| [DocTamper](https://github.com/qcf-568/DocTamper) | No clear repository-wide software licence found during the 2026-08-12 review | Paper/method and dataset-container reference only. No model/training source copied. The independent LMDB exporter interoperates with the documented `num-samples`, `image-%09d`, and `label-%09d` data keys. |
| [DanceText / DS-Net](https://github.com/qcf-568/DanceText) | No repository-wide software licence or model source present at commit `c4f177b4c1b1ab6e96bf01efbe37ff0f02396a9c` reviewed on 2026-08-12 | Method-level reference for hierarchical ViT + ConvNeXt-S ADN, ADN training roles, channel fusion, and classification/localization synergy. The local Synergy Denoising Decoder is a clean-room implementation because no DS-Net source was available. |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | MIT | Architectural reference for hierarchical windowed self-attention. The included compact ViT is a new stock-PyTorch implementation and does not copy this repository's source. |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | MIT | Architectural reference for the ADN's depthwise/pointwise residual blocks. No upstream source text or pretrained weights are included. |
| [Squeeze-and-Excitation Networks](https://github.com/hujie-frank/SENet) | Repository terms must be rechecked before source reuse | Method reference for channel attention between ViT and ADN features; the local module is independently implemented with stock PyTorch. |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause | Planned reference for later compound-degradation data generation; no code integrated yet. |
| [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) | CC-BY-4.0 shown by the repository | Benchmark protocol and future adapter target only; no source copied. Generic manifests can import compatible exported image/mask datasets. |

## Papers to cite

- Qu et al., “Detect Any AI-Counterfeited Text Image,” CVPR 2026.
- Liu et al., “Swin Transformer V2: Scaling Up Capacity and Resolution,” CVPR
  2022.
- Liu et al., “A ConvNet for the 2020s,” CVPR 2022.
- Hu, Shen, and Sun, “Squeeze-and-Excitation Networks,” CVPR 2018.
- Wong et al., “ADCD-Net: Robust Document Image Forgery Localization via
  Adaptive DCT Feature and Hierarchical Content Disentanglement,” ICCV 2025.
- Jiang, Zhang, and Timofte, “Towards Flexible Blind JPEG Artifacts Removal,”
  ICCV 2021.
- Qu et al., “Towards Robust Tampered Text Detection in Document Image: New
  Dataset and New Solution,” CVPR 2023.

Before publishing a paper, include the citations for DanceText/DS-Net, Swin,
ConvNeXt, squeeze-and-excitation, and any later-used components. Before copying
upstream source, re-check the exact commit and licence rather than relying on
this snapshot.
