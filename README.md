HyperMLP: Superpixel Prior and Feature Aggregated Perceptron Networks for Hyperspectral and LiDAR Hybrid Classification, TGRS, 2024
==
[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yuzhe Liu](https://github.com/lyz123-xidian), [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN),Wei Liu, [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).
***
Code for the paper: [HyperMLP: Superpixel Prior and Feature Aggregated Perceptron Networks for Hyperspectral and LiDAR Hybrid Classification].(https://ieeexplore.ieee.org/abstract/document/10401943).


<div align=center><img src="/Image/framework.png" width="80%" height="80%"></div>
Fig. 1: The overall of the proposed framework. It includes the superpixel segmentation prior information ”Super Token”, the LiDAR feature shuffle module ”FSM” and the feature fusion bilateral modulation strategy ”BMS”. The ”T” in the figure represents the feature dimension transpose. And the ’ERS’ stands for Entropy Rate Superpixel method.

Training and Test Process
--
1) Please prepare the training and test data as operated in the paper. The datasets are Houston2013, Trento, MUUFL Gulfport. The data is placed under the 'data' folder. The file format is tif.
2) Run "train_mixer.py" to to reproduce the HyperMLP results on Trento data set.

We have successfully tested it on Ubuntu 18.04 with PyTorch 1.12.0.

References
--
If you find this code helpful, please kindly cite:

[1]J. Li, Y. Liu, R. Song, W. Liu, Y. Li and Q. Du, "HyperMLP: Superpixel Prior and Feature Aggregated Perceptron Networks for Hyperspectral and LiDAR Hybrid Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-14, 2024, Art no. 5505614, doi: 10.1109/TGRS.2024.3355037.

Citation Details
--
BibTeX entry:
```
@ARTICLE{10401943,
  author={Li, Jiaojiao and Liu, Yuzhe and Song, Rui and Liu, Wei and Li, Yunsong and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HyperMLP: Superpixel Prior and Feature Aggregated Perceptron Networks for Hyperspectral and LiDAR Hybrid Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  keywords={Feature extraction;Hyperspectral imaging;Laser radar;Transformers;Data mining;Convolutional neural networks;Task analysis;Hybrid modality classification;hyperspectral image (HSI);light detection and ranging (LiDAR);MLP-Mixer},
  doi={10.1109/TGRS.2024.3355037}}
```
 
Licensing
--
Copyright (C) 2024 Yuzhe Liu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
