<div align="center">
<h2>Object Mask Module for Enhancing Multi-view 3D Occupancy Perception Performance Based on ViewFormer</h2>
</div>


- [2025/06]: Manuscript submitted (received on June 24, 2025).
- [2025/07]: Manuscript revised (on July 25, 2025).
- [2025/08]: Manuscript accepted (on August 20, 2025).
- [2025/10]: This work has been published in the Institute of Control, Robotics and Systems (ICROS, SCOPUS)


<p align="center">
    <a href='https://doi.org/10.5302/J.ICROS.2025.25.0168'><img src="https://img.shields.io/badge/Paper-PDF-blue?style=flat&#x26;logo=doi&#x26;logoColor=yello" alt="Paper PDF"></a>
</p>

🚀 This project base on [ViewFormerOcc](https://github.com/ViewFormerOcc/ViewFormer-Occ)
---

## Abstract

This study examines enhancing object detection by integrating an object-masking module into ViewFormer, a transformer-based model for 3D occupancy prediction from multi-view images. While ViewFormer effectively captures spatiotemporal information, it underperforms on small objects such as pedestrians and bicycles. To address this limitation, we designed a SegFormer-based object masking module that estimates object probabilities from BEV features and concatenates them as an additional feature channel. Experimental evaluations on the nuScenes dataset revealed an unexpected performance decline in overall metrics (mIoU, IoUgeo), particularly for small object detection. Subsequent analysis indicated weak mask activation and instability during initial training as key factors limiting the module’s effectiveness. These findings highlight the viability and constraints of object masking, underscoring the need for structural adjustments and improved training strategies to stabilize mask learning in future work.



## Keywords

autonomous driving, deep learning, 3D occupancy, viewformer, BEV representation, object masking


<img width="1275" height="304" alt="image" src="https://github.com/user-attachments/assets/7917cf87-49ec-45b4-a169-2de6a88e70f7" />


### Table 1. Comparison of model performance.
| Method | Training GPU | Test GPU | Training Time | Memory(G) | FPS↑ | FLOPs↓ | Params↓ | IoUgeo↑ | mIoU↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ViewFormer | 4090*2 | - | 5D 22h | 13.7 | 12.2 | 214.74 | 99.19 M | 71.03 | 41.37 |
| Try 1 (126) CNN | 4090*2 | 4090*2 | 7D 7h | 14.1 | 6.2 | 242.14 | 99.28 M | 62.48 | 25.45 |
| Try 2 (126) CNN | 4090*2 | 4090*2 | 6D 23h | 13.7 | 12.1 | 242.14 | 99.28 M | 62.76 | 25.73 |
| Ours (126) Seg | 4090*2 | 4090*2 | 9D 5h | 16.2 | 10.7 | 249.30 | 181.21 M | 64.89 | 28.76 |

### Table 2. Performance comparison of the U-Net prediction model.
| Method | Training GPU | Test GPU | Training Time | Memory(G) | FPS↑ | FLOPs↓ | Params↓ | IoUgeo↑ | mIoU↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ViewFormer | 4090*2 | 1/10 | - | 13.7 | - | - | 99.19 M | 65.56 | 32.23 |
| Try 3 (126) U-Net | 4090*2 | 1/10 | - | 15.1 | - | - | 130.30 M | 57.94 | 13.75 |

### Table 3. Early performance comparison when training the Seg model on a single GPU.
| Method | Training GPU | Test GPU | Training Time | Memory(G) | FPS↑ | FLOPs↓ | Params↓ | IoUgeo↑ | mIoU↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ViewFormer | 3090*1 | 1/10 | 22D | 14.6 | - | - | 99.19 M | 62.88 | 27.36 |
| Ours - single (126) Seg | 3090*1 | 1/10 | 32D | 14.9 | - | - | 181.21 M | 62.53 | 28.28 |

### Table 4. The performance comparison by each class.
*Each value represents the average IoU score for the corresponding class.*
| Method | others | barrier | bicycle | bus | car | const. veh. | motorcycle | pedestrian | traffic cone | trailer | truck | driv. surf. | other flat | sidewalk | terrain | manmade | vegetation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ViewFormer | 11.45 | 49.03 | 27.88 | 45.81 | 52.61 | 23.56 | 5.90 | 27.63 | 29.95 | 30.64 | 38.49 | 84.60 | 48.93 | 57.38 | 59.86 | 46.88 | 40.35 |
| Ours (126) Seg | 7.11 | 32.48 | 10.87 | 26.81 | 39.58 | 28.25 | 10.69 | 14.78 | 19.37 | 15.44 | 17.61 | 80.57 | 40.50 | 49.83 | 52.85 | 33.79 | 30.78 |
