# Cross-Camera Pedestrian Trajectory Retrieval Based on Linear Trajectory Manifolds

Welcome to refer to our other related papers:
[Person Trajectory Dataset/LocalCRF](https://github.com/zhangxin1995/PTD.git)|[Person Group Trajectory Dataset/GlocalCRF](https://github.com/zhangxin1995/PGTD.git)


## Description
This is the code implementation and related dataset repository for paper 'Cross-Camera Pedestrian Trajectory Retrieval Based on Linear Trajectory Manifolds'. We also integrated the previously proposed 'PTD' The `PGTD' benchmark is convenient for more researchers to compare.


## Abstract
The goal of pedestrian trajectory retrieval is to retrieve the trajectory of a given pedestrian image or video under the camera network. Existing methods of extracting cross-camera trajectory spatio-temporal models need to collect the trajectory data between each pair of cameras and calculate the distance between them, which has significant limitations. In this paper, we consider using only temporal information to extract cross-camera trajectories. We propose a Temporal Rotary Position Embedding(T-RoPE) method that embeds the temporal information of a single-camera tracklet into its visual feature, resulting in a new feature space. We refer to the structure of the cross-camera trajectory embedded in the new feature space as the trajectory manifold and refer to the linear trajectory manifold that satisfies the minimum cost and maximum flow constraint in the new feature space. The T-RoPE method is an implicit trajectory time model whose time model does not exist independently but directly embeds time information into trajectory visual features, so it does not require distance between camera pairs. Meanwhile, we also propose that the trajectory manifold in the feature space after T-RoPE encoding satisfies the minimum cost maximum flow constraint and is a linear trajectory manifold. To validate our method, we also collected a new pedestrian trajectory dataset, the Mall Trajectory Dataset. Finally, we conducted experiments on multiple datasets, and the results showed that our T-RoPE linear trajectory manifold extraction framework can be plug-and-play applied to different visual models, improving the accuracy of pedestrian retrieval.

## Framework
The Figure. 1 illustrates the framework of our method. We start by processing the collected surveillance videos through target detection and tracking to obtain single-camera trajectories. Next, we use a visual feature model to extract visual features for each single-camera trajectory. We then embed time information into these visual features to obtain new fused features, which constitute the trajectory manifold space. Subsequently, we employ a trajectory manifold extraction algorithm to extract cross-camera trajectories from the trajectory manifold space. Finally, we use the extracted cross-camera trajectories as a gallery to extract visual features for pedestrian retrieval tasks.

![Fig. 1: Framework](https://github.com/zhangxin1995/MTD/blob/master/images/framework.png)

## Pedestrian Trajectory Retrieval
![Fig. 2: Examples of pedestrian video retrieval, trajectory retrieval, and trajectory re-ranking.Each image represents the first frame of a tracklet. An image with a green border indicates that its identity is the same as the query, while an image with a red border means its identity differs from the query.](https://github.com/zhangxin1995/MTD/blob/master/images/example.png)
Fig. 2 shows the example retreival results of video based pedestrian retrieval, trajectory based pedestrian retrieval, and pedestrian trajectory reordering. It can be seen that trajectory based retrieval results not only include extra trajectory information, but also improve retrieval accuracy.



## Mall Trajectory Dataset(MTD)
![Table 1: The Mall Trajectory Dastaset.](https://github.com/zhangxin1995/MTD/blob/master/images/dataset.png)
![Fig. 2: The Mall Trajectory Dastaset.](https://github.com/zhangxin1995/MTD/blob/master/images/camera.png)
The Mall Trajectory Dataset was collected from a tens of thousands of square feet mall, from which we sampled 11 cameras for our experiments. In the MTD dataset, it consists of 527,066 images from 3845 individuals. Of these, the training set consists of 7060 single-camera tracks from 2750 individuals and the test set consists of 3404 single-camera tracks from 1095 individuals. Table 1 shows the comparison of MTD with existing pedestrian re-identification datasets, and it can be seen that our dataset includes not only more pedestrian images and more pedestrians but also a non-acting real-scene dataset, which is useful for investigating cross-camera spatio-temporal association. Fig. 3 shows captured pictures of the corresponding cameras in the Mall Trajectory Dataset. As can be seen in Fig. 3, our new dataset scenario is more complex compared to the existing public dataset. Firstly, the environment of our dataset is an indoor environment, with numerous access routes between each camera and no maps to refer to. Secondly, our dataset includes vertical lifts and escalators, such as cameras \#1, \#2, and \#3, and the running time of lifts is usually uncertain, which brings new challenges to spatio-temporal modeling. Finally, the difference in viewpoints in the scene and the occlusion by glass doors and railings raise the difficulty of the pedestrian retrieval task for this dataset.

### Download Url
You can download the dataset files used through [Google Drive](https://drive.google.com/file/d/1nEWPjyhZccolcE634XA3cXbDdAHKvN2w/view?usp=sharing).
You can download the model files used through [Google Drive](https://drive.google.com/file/d/13QSkNRTNhFFcUeOynpgEKr2ng9GxOQ0_/view?usp=sharing).


## Quick Start
To obtain consistent results with the experiments in the paper, please enter the following code and run it:
```
git clone https://github.com/zhangxin1995/MTD.git

#For the Mall Trajectory Dataset, PSTA + T-RoPE Temporal Model + Linear Trajectory Manifold
python main.py --yaml ./config/manifold_mall_td.yaml

#For the Mall Trajectory Dataset, PSTA + Gauss Temporal Model + Linear Trajectory Manifold
python main.py --yaml ./config/manifold_gauss_td.yaml

#For the Mall Trajectory Dataset, Resnet + T-RoPE Temporal Model + Linear Trajectory Manifold
python main.py --yaml ./config/manifold_mall_td_resnet.yaml

#For the Mall Trajectory Dataset, MGN + T-RoPE Temporal Model + Linear Trajectory Manifold
python main.py --yaml ./config/manifold_mall_td_mgn.yaml

#For the Mall Trajectory Dataset, TransReid + T-RoPE Temporal Model + Linear Trajectory Manifold
python main.py --yaml ./config/manifold_mall_td_trans.yaml

#For the Person Trajectory Dataset, ResNet + T-RoPE Temporal Model + Linear Trajectory Manifold
python main.py --yaml ./config/manifold_ptd_resnet.yaml

#For the Person Trajectory Dataset, MGN + T-RoPE Temporal Model + Linear Trajectory Manifold
python main.py --yaml ./config/manifold_ptd_mgn.yaml

#For the Person Trajectory Dataset, ResNet + LocalCRF
python main.py --yaml ./config/local_crf_resnet.yaml

#For the Person Group Trajectory Dataset, ResNet + LocalCRF
python main.py --yaml ./config/local_crf_pgtd_resnet.yaml 

#For the Person Group Trajectory Dataset, ResNet + GlobalCRF
python main.py --yaml ./config/local_crf_pgtd_resnet.yaml 
```

## Citation
If this code repository is helpful to you, please cite:

```
@ARTICLE{10176311,
  author={Zhang, Xin and Xie, Xiaohua and Lai, Jianhuang and Zheng, Wei-Shi},
  journal={IEEE Transactions on Image Processing}, 
  title={Cross-Camera Trajectories Help Person Retrieval in a Camera Network}, 
  year={2023},
  volume={32},
  number={},
  pages={3806-3820},
  keywords={Cameras;Pedestrians;Trajectory;Visualization;Legged locomotion;Spatiotemporal phenomena;Data models;Person retrieval;trajectory generation;person re-id;conditional random field},
  doi={10.1109/TIP.2023.3290515}}

@article{ZHANG2024127281,
title = {Person group detection with global trajectory extraction in a disjoint camera network},
journal = {Neurocomputing},
volume = {574},
pages = {127281},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.127281},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224000523},
author = {Xin Zhang and Xiaohua Xie and Li Wen and Jianhuang Lai}
}
```