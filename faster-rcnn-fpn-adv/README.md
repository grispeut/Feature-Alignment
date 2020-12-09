# FASTER-RCNN-FPN

The implementation of FASTER-RCNN-FPN is based on [fpn.pytorch](https://github.com/jwyang/fpn.pytorch) and [mmdetection](https://github.com/open-mmlab/mmdetection).

## Dependency

* Python 3.6 
* torch 1.6.0
* torchvision 0.7.0
* scipy 1.2.0

## Data Preparation
Follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data

* **PASCAL_VOC 0712**: 
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```
2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```

5. Follow similar steps to get PASCAL VOC 2012

## Pretrained Model

* **PASCAL_VOC 0712**: 

  Link: https://pan.baidu.com/s/1rGzK84IMA5eOPPZmU5U3Aw  password: 2buj


## Train

Training FASTER-RCNN-FPN using KDFA+SSFA on PASCAL_VOC 0712 can be done as follows:
```
python train.py --dataset pascal_voc_0712 --weights weights/voc_pretrained.npy --bs 8 --epochs 15 --adv --kdfa --ssfa
```

## Test

If you want to evlauate the clean AP and adversarial robustness on PASCAL_VOC 2007 test set, simply run
```
python test.py --dataset pascal_voc_0712 --weights weights/voc_pretrained.npy --step_size 0.03 --num_steps 1 
```

## Grad Cam

If you want to visualize the feature of Mth layer with Grad-CAM, simply run
```
python gradcam.py --dataset pascal_voc_0712 --weights weights/voc_pretrained.npy --source samples 
```

If you want to directly reproduce our results in our work, 
* get the models from Link: https://pan.baidu.com/s/1X9yQufBnQhJcVz5aficPvw  password: kgem

* visualize the feature trained with SSFA+KDFA, simply run
  ```
  python gradcam.py --dataset pascal_voc_0712 --weights voc_weights/KDFA+SSFA_PGD2.npy --source samples 
  ```
* visualize the feature trained with AT, simply run
  ```
  python gradcam.py --dataset pascal_voc_0712 --weights voc_weights/AT_PGD2.npy --source samples 
  ```
* visualize the feature trained with standard setting, simply run
  ```
  python gradcam.py --dataset pascal_voc_0712 --weights voc_weights/standard.npy --source samples 
  ```



## Citation

    @inproceedings{renNIPS15fasterrcnn,
    Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
    Title = {Faster {R-CNN}: Towards Real-Time Object Detection
             with Region Proposal Networks},
    Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
    Year = {2015}}


```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
```
@article{DBLP:journals/corr/SelvarajuDVCPB16,
	author    = {Ramprasaath R. Selvaraju and
	Abhishek Das and
	Ramakrishna Vedantam and
	Michael Cogswell and
	Devi Parikh and
	Dhruv Batra},
	title     = {Grad-CAM: Why did you say that? Visual Explanations from Deep Networks
	via Gradient-based Localization},
	journal   = {CoRR},
	volume    = {abs/1610.02391},
	year      = {2016},
	url       = {http://arxiv.org/abs/1610.02391},
	archivePrefix = {arXiv},
	eprint    = {1610.02391},
	timestamp = {Mon, 13 Aug 2018 16:46:58 +0200},
	biburl    = {https://dblp.org/rec/journals/corr/SelvarajuDVCPB16.bib},
	bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

    

    

