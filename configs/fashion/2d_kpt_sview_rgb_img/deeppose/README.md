# Deeppose: Human pose estimation via deep neural networks

## Introduction

<!-- [ALGORITHM] -->

<details>
<summary align="right">DeepPose (CVPR'2014)</summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

</details>

DeepPose first proposes using deep neural networks (DNNs) to tackle the problem of keypoint detection.
It follows the top-down paradigm, that first detects the bounding boxes and then estimates poses.
It learns to directly regress the fashion keypoint coordinates.
