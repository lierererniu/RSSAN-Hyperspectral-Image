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

运行方法  
python main.py


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
  
  PU：OA, AA, kappa: [0.9916494087781415, 0.9869853556897955, 0.9889365386137344]  
       each_acc [98.64, 99.77, 95.03, 99.44, 99.79, 99.89, 97.85, 98.33, 99.55]  
       
  KSC：OA, AA, kappa: [0.8999450247388675, 0.8738564837073592, 0.8883809689177758]  
       each_acc [99.06, 64.5, 97.19, 57.14, 85.71, 94.38, 100.0, 99.67, 100.0, 64.18, 100.0, 74.64, 99.54]  
       
  IN：OA, AA,kappa: [0.9861789752896831, 0.9721097015738396, 0.9842439139710597]  
       each_acc [100.0, 98.4, 98.97, 93.94, 100.0, 99.41, 94.74, 100.0, 78.57, 97.5, 98.49, 98.79, 100.0, 99.55, 97.03, 100.0]  
       
 问题：  
  KSC数据集的表现很差  
  
  
  # 2021.8.19  
  将论文中的消融实验加上了，自行运行，对比结果。  
