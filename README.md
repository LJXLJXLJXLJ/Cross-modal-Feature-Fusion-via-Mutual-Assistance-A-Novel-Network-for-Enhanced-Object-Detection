# Cross-modal Feature Fusion via Mutual
Assistance: A Novel Network for Enhanced
Object Detection

## Installation 
Python>=3.6.0 is required with all requirements.txt installed including PyTorch>=1.7 (The same as yolov5 https://github.com/ultralytics/yolov5 ).


  
#### Install requirements


pip install -r requirements.txt


## Dataset


-[LLVIP]  [download](https://github.com/bupt-ai-cz/LLVIP)

-[M3FD]  [download]
The details of the dataset download and partitioning can be found in this paper：

Mingjian Liang, Junjie Hu, Chenyu Bao, Hua Feng, Fuqin
Deng, and Tin Lun Lam. Explicit attention-enhanced fu-
sion for rgb-thermal perception tasks. IEEE Robotics Autom.
Lett., 8(7):4060–4067, 2023. 


You need to convert all annotations to YOLOv5 format.

Refer: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

## Run
#### Download the pretrained weights
yolov5 weights (pre-train) 

-[yolov5s] [google drive](https://drive.google.com/file/d/1UGAsaOvV7jVrk0RvFVYL6Vq0K7NQLD8H/view?usp=sharing)

-[yolov5m] [google drive](https://drive.google.com/file/d/1qB7L2vtlGppGjHp5xpXCKw14YHhbV4s1/view?usp=sharing)

-[yolov5l] [google drive](https://drive.google.com/file/d/12OFGLF73CqTgOCMJAycZ8lB4eW19D0nb/view?usp=sharing)

-[yolov5x] [google drive](https://drive.google.com/file/d/1e9xiQImx84KFQ_a7XXpn608I3rhRmKEn/view?usp=sharing)

training  weights ：


https://pan.baidu.com/s/1GDZQV4ou9QpQz-Trxs3c0w 提取码: k8hm


#### Change the data cfg
".\data\multispectral"





### Train Test and Detect
train: ``` python train.py```
The train.py file is still being organized, and we will upload the complete code shortly.
test: ``` python test.py```

detect: ``` python detect_twostream.py```


#### References

https://github.com/ultralytics/yolov5

  
