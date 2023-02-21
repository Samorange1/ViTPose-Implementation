# Implementation of ViTPose

This is our implementation of ViTPose algorithm for human pose estimation using a Vision Transformer. The original code can be found at this [GitHub repository](https://github.com/ViTAE-Transformer/ViTPose).


We have taken pretrained weights from the [MAE Pretrained Model](https://1drv.ms/u/s!AimBgYV7JjTlgccZeiFjh4DJ7gjYyg?e=iTMdMq)

We then trained our model on the MS COCO dataset for about 100 epochs with 90,000 images.
Given our hardware limitations, we still achieved a commendable accuracy of 50%

The following images are some of our outputs:
Note: These images show the keypoint detections of the human skeleton. We have tried to detect 17 keypoints.

![ViTPose Output 1](https://github.com/Samorange1/ViTPose-Implementation/blob/main/Output/Output_img1.jpg)
![ViTPose Output 2](https://github.com/Samorange1/ViTPose-Implementation/blob/main/Output/Output_img2.jpg)
![ViTPose Output 3](https://github.com/Samorange1/ViTPose-Implementation/blob/main/Output/Output_img3.jpg)
