## Dataset 
The overall directory structure should be:

```
│Point-LGMask/
├──cfgs/
├──datasets/
├──data/
│  ├── ModelNet
│  ├── ModelNetFewshot
│  ├── S3DIS
│  ├── ScanObjectNN
│  ├── ScanObjectNN_Fewshot_like_crossPoint
│  ├── ScanObjectNN_cross_point
│  ├── ScanObjectNN_shape_names.txt
│  ├── ShapeNet55-34
│  ├── shapenet_synset_dict.json
│  ├── shapenetcore_partanno_segmentation_benchmark_v0_normal
├──.......
```

**ModelNet40 Dataset:**
```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```
* Download: The data can be downloaded from [Point-BERT](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or from the [official website](https://modelnet.cs.princeton.edu/#) and processed by yourself.

**ModelNet Few-shot Dataset:**  
```
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```
* Download: The data can be downloaded from [Point-BERT](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.



**ScanObjectNN Dataset:**  
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
* Download: The data can be downloaded from [official website](https://hkust-vgd.github.io/scanobjectnn/).

**ScanObjectNN Few-shot Dataset:**  
```
│ScanObjectNN_Fewshot_like_crossPoint/
├── 10way_10shot
├── 10way_1shot
├── 10way_20shot
├── 10way_3shot
├── ...
```
* The data can be downloaded from [here](https://drive.google.com/file/d/1RgAmPOavfvm-7P4FxK2LtfpWKoIYanFF/view?usp=sharing) .
* Or you can generate your own few-shot learning split on ScanObjectNN Dataset. 
First, download [ScanObjectNN](https://drive.google.com/file/d/1r3nJ6gEu6cL59h7cIfMrBeY5p1P_JiIV/view?usp=drive_link)  from [CrossPoint](https://github.com/MohamedAfham/CrossPoint), and put it to ```./data/ScanObjectNN_cross_point```.
like:
  ```
  data/ScanObjectNN_cross_point
  ├── label_name.txt
  ├──  main_split
  ```
  Then, run ```python ./datasets/generate_few_shot_data_SacnObjectNN_like_crossPoint.py```. 
  Last, you can see your own split on ```./data/ScanObjectNN_Fewshot_like_crossPoint```.

**ShapeNet55/34 Dataset:**  
```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-35/
│  ├── train.txt
│  └── test.txt
```
* Download: The data can be downloaded from [Point-BERT](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.


**ShapeNetPart Dataset:**  
```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```
* Download: The data can be downloaded from [Point-BERT](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

**S3DIS Dataset:**

``` 
|S3DIS/
|-- S3DIS
|   |-- ReadMe.txt
|   |-- Stanford3dDataset_v1.2-001.mat
|   |-- Stanford3dDataset_v1.2-003.zip
|   |-- Stanford3dDataset_v1.2_Aligned_Version
|   |   |-- Area_1
|   |   |   |-- WC_1
|   |   |   |   |-- Annotations
|   |   |   |   |   |-- ceiling_1.txt
|   |   |   |   |   |-- ceiling_2.txt
│   |   |   |.......
│-- stanford_indoor3d
│   |-- Area_1_WC_1.npy
│   |-- Area_1_conferenceRoom_1.npy
│   |-- Area_1_conferenceRoom_2.npy
│   |-- Area_1_copyRoom_1.npy
│   |.......
```

Please prepare the dataset following [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch):
download the `Stanford3dDataset_v1.2_Aligned_Version` from [here](http://buildingparser.stanford.edu/dataset.html), and get the processed `stanford_indoor3d` with:

```shell
cd datasets
python collect_indoor3d_data.py
```