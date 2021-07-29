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
  test batch_size: 16 
  lr: IN,PU 0.0003 KSC 0.0001  
  optimizer:RMSprop  
  depth:PU 8 IN,KSC 32  
  kernel_size: 3  
  
  PU：OA, AA, kappa: [0.9910147638452802, 0.9845195729019799, 0.988088815434903]  
  KSC：OA, AA, kappa: [0.8999450247388675, 0.8738564837073592, 0.8883809689177758]  
       each_acc [99.06, 64.5, 97.19, 57.14, 85.71, 94.38, 100.0, 99.67, 100.0, 64.18, 100.0, 74.64, 99.54]  
  IN：OA, AA, kappa: [0.9810135418120899, 0.9543826383543743, 0.9783469945643659]
       each_acc [90.32, 97.3, 98.28, 89.09, 99.11, 99.22, 89.47, 98.8, 78.57, 98.23, 98.49, 97.34, 100.0, 100.0, 95.91, 96.88]  
  
