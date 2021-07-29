# RSSAN-Hyperspectral-Image
Residual Spectral–Spatial Attention Network for Hyperspectral Image Classification 2020 IEEE TGRS  

10.1109/TGRS.2020.2994057  

https://ieeexplore.ieee.org/abstract/document/9103247  

论文代码复现  

The pytorch framework used in this code. Instead of the TensorFlow framework used in the article, modify it yourself if necessary.  

本代码采用的pytorch框架。而不是文章采用的TensorFlow框架，如有需要自行修改。  
RSSAN-Hyperspectral-Image  
  --Dateset  
  --function  
  --model  
  --resulit  
  --main.py  
  --README.md  
  --train.log  

environment：  
  python 3.8.5  
  numpy 1.19.2  
  scikit-learn 0.23.2  
  tensorflow 2.5.0  
  torch 1.9.0  
  ....  
  
![image](https://user-images.githubusercontent.com/41353851/127426989-d1af6823-02f4-4425-85a4-67c0ae5abcf1.png)

The running results are saved in the result folder.  

运行结果均保存在result文件夹中。  


Parameter setting:  
参数设置：  

  epoch: 200  
  patch_size: 17  
  train batch_size: 16  
  test batch_size: 100  
  lr: IN,PU 0.0003 KSC 0.0001  
  optimizer:RMSprop  
  depth:PU 8 IN,KSC 32  
  kernel_size: 3  
  
