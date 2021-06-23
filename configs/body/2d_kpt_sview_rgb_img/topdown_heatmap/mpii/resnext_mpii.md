<!-- [BACKBONE] -->

<details>
<summary align="right">ResNext (CVPR'2017)</summary>

```bibtex
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
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
| [pose_resnext_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/resnext152_mpii_256x256.py) | 256x256 | 0.887 | 0.294 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_mpii_256x256-df302719_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_mpii_256x256_20200927.log.json) |
