<!-- [ALGORITHM] -->

<details>
<summary align="right">Associative Embedding (NIPS'2017)</summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
```

</details>

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

<!-- [DATASET] -->

<details>
<summary align="right">MHP (ACM MM'2018)</summary>

```bibtex
@inproceedings{zhao2018understanding,
  title={Understanding humans in crowded scenes: Deep nested adversarial learning and a new benchmark for multi-human parsing},
  author={Zhao, Jian and Li, Jianshu and Cheng, Yu and Sim, Terence and Yan, Shuicheng and Feng, Jiashi},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={792--800},
  year={2018}
}
```

</details>

Results on MHP v2.0 validation set without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w48](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_mhp_512x512.py)  | 512x512 | 0.583 | 0.895 | 0.666 | 0.656 | 0.931 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_mhp_512x512-85a6ab6f_20201229.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_mhp_512x512_20201229.log.json) |

Results on MHP v2.0 validation set with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w48](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_mhp_512x512.py)  | 512x512 | 0.592 | 0.898 | 0.673 | 0.664 | 0.932 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_mhp_512x512-85a6ab6f_20201229.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_mhp_512x512_20201229.log.json) |
