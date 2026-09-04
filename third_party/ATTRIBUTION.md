# Attribution and reuse boundary

DeepDocForgery is a new integration, not a fork of one upstream project.

| Upstream work | Use in this release |
|---|---|
| [ADCD-Net](https://github.com/KahimWong/ADCD-Net) | Quantization-table extraction and adaptive-DCT concepts. Its MIT notice is retained in `ADCD-Net-LICENSE`. |
| [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | Method reference for JPEG-quality and double-compression estimation; no restoration source is copied. |
| [DocTamper](https://github.com/qcf-568/DocTamper) | Dataset protocol and LMDB interoperability. No upstream model source is copied. |
| [MIDV-DM](https://github.com/SmartEngines/midv-dm) | Dataset contract for authentic/forged photographs, masks, and annotations. |
| [DanceText / DS-Net](https://github.com/qcf-568/DanceText) | Method reference for the ADN and classification/localization synergy. The decoder is independently implemented. |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | CUDA profiles obtain pretrained models through `timm`; weights are not included. |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | Architectural reference for independently implemented ADN blocks. |

Before publication or redistribution, re-check every dataset's current terms
and cite the precise versions used. Dataset bytes remain governed by their
owners' terms independently of this repository's GPL licence.
