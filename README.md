# H<sup>2</sup>FA R-CNN
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/h2fa-r-cnn-holistic-and-hierarchical-feature/weakly-supervised-object-detection-on-2)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-2?p=h2fa-r-cnn-holistic-and-hierarchical-feature)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/h2fa-r-cnn-holistic-and-hierarchical-feature/weakly-supervised-object-detection-on-1)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-1?p=h2fa-r-cnn-holistic-and-hierarchical-feature)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/h2fa-r-cnn-holistic-and-hierarchical-feature/weakly-supervised-object-detection-on-comic2k)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-comic2k?p=h2fa-r-cnn-holistic-and-hierarchical-feature)


This branch includes the official Detectron2 (Pytorch) implementation and pre-trained models for our paper:

[H<sup>2</sup>FA R-CNN: Holistic and Hierarchical Feature Alignment for Cross-domain Weakly Supervised Object Detection, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_H2FA_R-CNN_Holistic_and_Hierarchical_Feature_Alignment_for_Cross-Domain_Weakly_CVPR_2022_paper.html)

> **Abstract:** *Cross-domain weakly supervised object detection (CDWSOD) aims to adapt the detection model to a novel target domain with easily acquired image-level annotations. How to align the source and target domains is critical to the CDWSOD accuracy. Existing methods usually focus on partial detection components for domain alignment. In contrast, this paper considers that all the detection components are important and proposes a Holistic and Hierarchical Feature Alignment (H<sup>2</sup>FA) R-CNN. H<sup>2</sup>FA R-CNN enforces two image-level alignments for the backbone features, as well as two instance-level alignments for the RPN and detection head. This coarse-to-fine aligning hierarchy is in pace with the detection pipeline, i.e., processing the image-level feature and the instance-level features from bottom to top. Importantly, we devise a novel hybrid supervision method for learning two instance-level alignments. It enables the RPN and detection head to simultaneously receive weak/full supervision from the target/source domains. Combining all these feature alignments, H<sup>2</sup>FA R-CNN effectively mitigates the gap between the source and target domains. Experimental results show that H<sup>2</sup>FA R-CNN significantly improves cross-domain object detection accuracy and sets new state of the art on popular benchmarks.*

<div align="center">
  <img src="docs/intro.png" width=700/>
</div>

## Installation
### Requirements
* Linux with CUDA ≥ 9.2, gcc & g++ ≥ 4.9, Python ≥ 3.6
* PyTorch ≥ 1.4 and torchvision that matches the PyTorch installation. Note, please check PyTorch version matches that is required by Detectron2
* Detectron2 v0.2, following [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

### Build H<sup>2</sup>FA R-CNN
```
git clone https://github.com/XuYunqiu/H2FA_R-CNN.git
cd H2FA_RCNN
python -m pip install -e .
```


## Data Preparation
* Download PASCAL VOC 07 and 12 datasets from [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

* Download Clipart, Watercolor and Comic datasets from [cross domain detection](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets)

* Download Cityscapes and Fogy Cityscapes datasets from [da-faster-rcnn-PyTorch](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch)

For a few datasets that H<sup>2</sup>FA R-CNN natively supports, the datasets are assumed to exist in a directory called "datasets/", under the directory where you launch the program. They need to have the following directory structure:
```
DETECTRON2_DATASETS/
├── results
└── {VOC2007,VOC2012,Clipart,Watercolor,Comic,fogycityscapes}/
    ├── Annotations/
    ├── ImageSets/
    └── JPEGImages/
```
You can set the location for builtin datasets by export DETECTRON2_DATASETS=/path/to/datasets. If left unset, the default is ./datasets relative to your current working directory.

Note, the size of some target domain images is inconsistent with that provided in their annotations. You may need to resize these images according to the annotations. These images are listed in [image_list](mismatch_anno_list.txt).


## Getting Started

### Training & Evaluation in Command Line
To train a model, run 
```
cd tools/
python train_net.py --num-gpus 2 \
  --config-file ../configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_clipartall.yaml
```
The configs are made for 2-GPU training. To train on 1 GPU, you may need to change some parameters, *e.g.*:
```
python train_net.py \
  --config-file ../configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_clipartall.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

To evaluate the trained models, use
```
./train_net.py \
  --config-file ../configs/PascalVOC-Detection/cdod_faster_rcnn_R_101_DC5.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## Main Results with Pre-trained Models
Pre-trained models with R101-DC5 backbone on six datasets are available. All our models are trained on 2 NVIDIA V100-32G GPUs. 


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">iterations</th>
<th valign="bottom"><br/>mAP</th>
<th valign="bottom">pre-train model</th>
<th valign="bottom">metrics</th>
<!-- TABLE BODY -->

<tr><td align="left"><a href="configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_clipartall.yaml">VOC -> Clipart<sub>all<sub></a></td>
<td align="center">36k</td>
<td align="center">69.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1O9wMSB9-EHOJgv6HQ4P76aHZmVLHfOnT/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1G3-F7zjgwBeRZPTq1gg04Q?pwd=gob0">Baidu Pan</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1NJzcXrwGIa3pqti7mBV0ix9KK0wqNM4W/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/143Y103lRMVdBGXTGi085lg?pwd=l2ba">Baidu Pan</a></td>
</tr>

<tr><td align="left"><a href="configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_cliparttest.yaml">VOC -> Clipart<sub>test<sub></a></td>
<td align="center">24k</td>
<td align="center">55.3</td>
<td align="center"><a href="https://drive.google.com/file/d/1BjGMcSZw1CgkeBlG4cPepWLfIelI4bv3/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1zUyVO1-FaJ1ZZ-b7GjYSLA?pwd=ed2n">Baidu Pan</a></td>
<td align="center"><a href="https://drive.google.com/file/d/13AgH4-K_xgOAe_emtf9LIAcDw2mZt1ua/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/11hTXmg7gUskqYbBDEw_1AA?pwd=56v5">Baidu Pan</a></td>
</tr>

<tr><td align="left"><a href="configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_watercolor.yaml">VOC -> Watercolor</a></td>
<td align="center">24k</td>
<td align="center">59.9</td>
<td align="center"><a href="https://drive.google.com/file/d/1DNffSjIJnx5awj6SUaSPSUOJph2rXKzi/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/17FSO4ZCddVL4C8j3Nb7oaQ?pwd=2hut">Baidu Pan</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1P1nsLEYR2utDyAcZm6vzWMXr38URF_IU/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1JHpjT3d2_63GXtWrPYPi-A?pwd=h86l">Baidu Pan</a></td>
</tr>

<tr><td align="left"><a href="configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_comic.yaml">VOC -> Comic</a></td>
<td align="center">24k</td>
<td align="center">46.4</td>
<td align="center"><a href="https://drive.google.com/file/d/12FQIIS_m-dS3kQP69RMdUTKh5puiwzQm/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/14zNbzmJOncjltytbZPeiwQ?pwd=oo1c">Baidu Pan</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1TpJ6QAkkmKNdTYhXjBbc6r0Am10iBUso/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1q1iiHEaMC7ymjMUpxAtZsA?pwd=rp4c">Baidu Pan</a></td>
</tr>

<tr><td align="left"><a href="configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_watercolorextra.yaml">VOC -> Watercolor<sub>extra<sub></a></td>
<td align="center">36k</td>
<td align="center">62.6</td>
<td align="center"><a href="https://drive.google.com/file/d/10Y60uoWiZ0zAoHXBvqXbe49SdKrQu0OR/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1pS0zjGVLBEJbOMVAKR5xMQ?pwd=4nvq">Baidu Pan</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1gGkxMuxONu2EQ47OItvZiSxjaReDRqpx/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1tu-P5ZqdTkzh_txDBpNhzg?pwd=auiv">Baidu Pan</a></td>
</tr>

<tr><td align="left"><a href="configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_comicextra.yaml">VOC -> Comic<sub>extra<sub></a></td>
<td align="center">36k</td>
<td align="center">53.0</td>
<td align="center"><a href="https://drive.google.com/file/d/1Q4y-SpSG_UqtkEuXrjnfv4u8JFW4L5OW/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1j4SwEWGD6_Cww7lg5haj2Q?pwd=948v">Baidu Pan</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1cz6iZG3rLlgBW6emGuxZ7zdmv2et5zKe/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/11A21nT463cYepfFvZXSGmw?pwd=oo04">Baidu Pan</a></td>
</tr>

<tr><td align="left"><a href="configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_foggycityscapes.yaml">Cityscapes -> Foggy Cityscapes</a></td>
<td align="center">24k</td>
<td align="center">47.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1SyaGUQsMrFwDUMh8kKU6vDubqFvM220v/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1b_U-R8jubKtGj2ySSJ-DbQ?pwd=1hur">Baidu Pan</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1nFol3Ul2PqXe3PW90M65YS1lGXW6nQp_/view?usp=sharing">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1dVcVVgxUebUrbsqv1Ovxzg?pwd=uo38">Baidu Pan</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>


## TODO
- [x] Detectron2 implementation (this branch)
- [x] PaddleDetection implementation ([ppdet branch](https://github.com/XuYunqiu/H2FA_R-CNN/tree/ppdet))
- [ ] Code refactoring to make Detectron2 a third party


## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```BibTeX
@inproceedings{xu2022h2fa,
  title={{H$^2$FA R-CNN}: Holistic and Hierarchical Feature Alignment for Cross-domain Weakly Supervised Object Detection},
  author={Xu, Yunqiu and Sun, Yifan and Yang, Zongxin and Miao, Jiaxu and Yang, Yi},
  booktitle={Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  pages={14329-14339},
}
```

## License
This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
We build the project based on [Detectron2](https://github.com/facebookresearch/detectron2) and [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection). Thanks for their contributions.

## Contact
If you have any questions, please drop me an email: imyunqiuxu@gmail.com

