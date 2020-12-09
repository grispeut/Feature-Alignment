# YOLO-V3

The implementation of YOLO-V3 is based on [yolov3](https://github.com/ultralytics/yolov3) and [YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning).

## Dependency

* Python 3.6 
* torch 1.6.0

## Data Preparation
Please prepare dataset under the **data** directory 

* **PASCAL_VOC 0712**: 

  You can follow the instructions in [Convert2Yolo](https://github.com/ssaru/convert2Yolo), or download from link(Link: https://pan.baidu.com/s/1NzZQPNNSgdlKyTltESJzUA  password: 1ave)

* **COCO**: 
  ```
  cd data && bash get_coco_dataset.sh
  ```
## Pretrained Model

* **PASCAL_VOC 0712**: 

  Link: https://pan.baidu.com/s/1o0WcLRmkTaCQ3AWRem7hGw  password: nour

* **COCO**: 

  Link: https://pan.baidu.com/s/1dLAIb98Xgp9oB9jnPJKCTw  password: o0jo

## Train

* Training YOLO-V3 using KDFA+SSFA on PASCAL_VOC 0712 can be done as follows:
  ```
  python train.py --data data/voc.data --cfg cfg/yolov3-voc.cfg --weights weights/voc_pretrained.weights --bs 64 --epochs 30 --adv --kdfa --ssfa --kdfa_weights weights/voc_pretrained.weights
  ```

* Training YOLO-V3 using KDFA+SSFA on COCO can be done as follows:
  ```
  python train.py --data data/coco.data --cfg cfg/yolov3.cfg  --weights weights/coco_pretrained.weights --bs 64 --epochs 15 --adv --kdfa --ssfa --kdfa_weights weights/coco_pretrained.weights
  ```

## Test

* If you want to evlauate the clean AP and adversarial robustness on PASCAL_VOC 2007 test set, simply run
  ```
  python test.py --data data/voc.data --cfg cfg/yolov3-voc.cfg --weights weights/voc_pretrained.weights --step_size 0.03 --num_steps 1 
  ```

* If you want to evlauate the clean AP and adversarial robustness on COCO minival set, simply run
  ```
  python test.py --data data/coco.data --cfg cfg/yolov3.cfg --weights weights/coco_pretrained.weights --step_size 0.03 --num_steps 1 
  ```


    

    

