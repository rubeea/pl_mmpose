<!-- [ALGORITHM] -->

<details>
<summary align="right">HRNet (CVPR'2019)</summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right">Albumentations (Information'2020)</summary>

```bibtex
@article{buslaev2020albumentations,
  title={Albumentations: fast and flexible image augmentations},
  author={Buslaev, Alexander and Iglovikov, Vladimir I and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A},
  journal={Information},
  volume={11},
  number={2},
  pages={125},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
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
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [coarsedropout](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_coarsedropout.py)  | 256x192 | 0.753 | 0.908 | 0.822 | 0.806 | 0.946 | [ckpt](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_coarsedropout-0f16a0ce_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_coarsedropout_20210320.log.json) |
| [gridmask](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_gridmask.py)  | 256x192 | 0.752 | 0.906 | 0.825 | 0.804 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_gridmask-868180df_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_gridmask_20210320.log.json) |
| [photometric](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_photometric.py)  | 256x192 | 0.753 | 0.909 | 0.825 | 0.805 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_photometric-308cf591_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_photometric_20210320.log.json) |
