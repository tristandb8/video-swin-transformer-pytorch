# Video-Swin-Transformer-Pytorch
This repo is a simple usage of the official implementation ["Video Swin Transformer"](https://github.com/SwinTransformer/Video-Swin-Transformer).

![teaser](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/figures/teaser.png)


### Prepare
```
$ git clone https://github.com/haofanwang/video-swin-transformer-pytorch.git
$ cd video-swin-transformer-pytorch
$ mkdir checkpoints && cd checkpoints
$ wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth
$ cd ..
```
Please refer to [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) and download other checkpoints.

### Inference

To run main file:
```
python VidSwinMain.py --logFolder nameoffolderforlogstobesaved
```
Note that this code isn't yet prepared for other users.
This repo doesn't include the dataloader files either. Waiting upon approval of paper by IROS conference


## Acknowledgement
The code is adapted from the official [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) repository. This project is inspired by [swin-transformer-pytorch](https://github.com/berniwal/swin-transformer-pytorch), which provides the simplest code to get started.


## Citation
If you find our work useful in your research, please cite:

```
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}

@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
