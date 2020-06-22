# **Car Detection for Autonomous Driving** 

### The project implementation comprises of the following steps.

* Input image (608, 608, 3)
* The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
* After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
  * Each cell in a 19x19 grid over the input image gives 425 numbers.
  * 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
  * 85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw)(pc,bx,by,bh,bw)  has 5 numbers, and 80 is the number of classes we'd like to detect
* Then we select only few boxes based on:
  * Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
  * Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
* This gives the YOLO's final output.

![](output/test.jpg)

### YOLO
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

#### Model details
#### Inputs and outputs
- The **input** is a batch of images, and each image has the shape (m, 608, 608, 3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers $(p_c, b_x, b_y, b_h, b_w, c)$ as explained above. If you expand $c$ into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 

#### Anchor Boxes
* Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes.  For this assignment, 5 anchor boxes were chosen for you (to cover the 80 classes), and stored in the file './model_data/yolo_anchors.txt'
* The dimension for anchor boxes is the second to last dimension in the encoding: (m, n_H,n_W,anchors,classes).
* The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).  

#### Encoding
Let's look in greater detail at what this encoding represents. 

![](rm_images/architecture.png)
<p align="center"> Figure: Encoding architecture for YOLO. </p>

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.
Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.
For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).

<div align="center">
<img src="rm_images/flatten.png" style="width:500px;height:400;">
<p align="center"> Figure: Flattening the last two last dimensions. </p>
</div>

### Non-max suppression

Even after filtering by thresholding over the class scores, you still end up with a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). 
Specifically, following steps are carried out: 
- Get rid of boxes with a low score (meaning, the box is not very confident about detecting a class; either due to the low probability of any object, or low probability of this particular class).
- Select only one box when several boxes overlap with each other and detect the same object.
<div align="center">
<img src="rm_images/non-max-suppression.png" style="width:500px;height:400;">
<p align="center"> Figure: In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probability) of the 3 boxes. </p>
</div>

Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU.

<div align="center">
<img src="rm_images/iou.png" style="width:500px;height:400;">
<p align="center"> Figure: Definition of "Intersection over Union". <p>
</div>

If you were to run your session in a for loop over all your images. Here's what you would get:

![](rm_images/pred_video_compressed2.mp4)
Predictions of the YOLO model on pictures taken from a camera while driving around the Silicon Valley <br> Thanks [drive.ai][https://www.drive.ai/] for providing this dataset!

### References: ### 
The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's GitHub repository. The pre-trained weights used in this exercise came from the official YOLO website. 
- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
- Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
- Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- The official YOLO website (https://pjreddie.com/darknet/yolo/) 
