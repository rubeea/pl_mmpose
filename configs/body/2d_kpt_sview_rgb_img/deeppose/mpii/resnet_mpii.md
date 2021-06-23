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

<!-- [BACKBONE] -->

<details>
<summary align="right">ResNet (CVPR'2016)</summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right">MPII (CVPR'2014)</summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [deeppose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/deeppose/mpii/res50_mpii_256x256.py) | 256x256 | 0.825 | 0.174 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256-c63cd0b6_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_20210203.log.json) |
| [deeppose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/deeppose/mpii/res101_mpii_256x256.py) | 256x256 | 0.841 | 0.193 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_mpii_256x256-87516a90_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_mpii_256x256_20210205.log.json) |
| [deeppose_resnet_152](/configs/body/2d_kpt_sview_rgb_img/deeppose/mpii/res152_mpii_256x256.py) | 256x256 | 0.850 | 0.198 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_mpii_256x256-15f5e6f9_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_mpii_256x256_20210205.log.json) |
