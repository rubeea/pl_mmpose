<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
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
<summary align="right">COCO (ECCV'2014)</summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py)  | 256x192 | 0.718 | 0.898 | 0.795 | 0.773 | 0.937 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_20200709.log.json) |
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_384x288.py)  | 384x288 | 0.731 | 0.900 | 0.799 | 0.783 | 0.931 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288-e6f795e9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_20200709.log.json) |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192.py) | 256x192 | 0.726 | 0.899 | 0.806 | 0.781 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_20200708.log.json) |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_384x288.py) | 384x288 | 0.748 | 0.905 | 0.817 | 0.798 | 0.940 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288-8c71bdc9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_20200709.log.json) |
| [pose_resnet_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192.py) | 256x192 | 0.735 | 0.905 | 0.812 | 0.790 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_20200709.log.json) |
| [pose_resnet_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_384x288.py) | 384x288 | 0.750 | 0.908 | 0.821 | 0.800 | 0.942 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_20200709.log.json) |
