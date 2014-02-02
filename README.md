# [Car Detection](http://carlosmccosta.github.io/Car-Detection/)


## Overview
This project focus on the detection and recognition of cars in different perspective views and has the following associated paper:

[Multiview object recognition using Bag of Words approach](https://github.com/carlosmccosta/Car-Detection/raw/master/Report/Multiview%20object%20recognition%20using%20Bag%20of%20Words%20approach.pdf)


**Abstract:**
Multiview object detection and classification plays a 
critical  role  in  robust  image  recognition  systems,  and  can  be 
applied  in  a  multitude  of  applications,  ranging  from  simple 
monitoring  to advanced tracking. In this paper it is analyzed the 
usage of the Bag of Words model to efficiently detect and recognize 
objects that can appear in different  scales, orientations and even 
from  different  perspective  views.  This  approach  relies  in  image 
analysis  techniques,  such  as  feature  detection,  description  and 
clustering, in order to be able to recognize the target object even if 
it  is  present  in  cluttered  environments.  For  supporting  the 
recognition  in  different  perspective  views,  machine  learning 
techniques  are  used  to build  a  model  of the target  objects. This 
model  can  then  be  employed  to  successfully  recognize  if  an 
instance of the  target object is present in  an  image.  For pinpointing 
the location  of the target object, a sliding window method is used
in conjunction with dynamic thresholding. The recognition system 
was  tested  with  several  configurations  of  feature  detectors, 
descriptors  and classifiers, and achieved an accuracy of 87% when 
recognizing cars from 177 test images.

## Results

![Fig. 1 - Effect of preprocessing (right) in the original image 
(left)](https://raw2.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/1%20-%20Effect%20of%20preprocessing%20(right)%20in%20the%20original%20image%20(left).png)
Fig. 1 - Effect of preprocessing (right) in the original image 
(left)



![Fig. 2 - Target objects ground truth masks](https://raw2.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/2%20-%20Target%20objects%20ground%20truth%20masks.png)
Fig. 2 - Target objects ground truth masks



![Fig. 3 - Results obtained with STAR detector, SIFT extractor, FLANN matcher and ANN classifier](https://raw2.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/3%20-%20Results%20obtained%20with%20STAR%20detector,%20SIFT%20extractor,%20FLANN%20matcher%20and%20ANN%20classifier.png)
Fig. 3 - Results obtained with STAR detector, SIFT extractor, FLANN matcher and ANN classifier



![Fig. 4 - Results with partially occluded objects obtained with STAR detector, SURF extractor, FLANN matcher and SVM classifier](https://raw2.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/4%20-%20Results%20obtained%20with%20STAR%20detector,%20SURF%20extractor,%20FLANN%20matcher%20and%20SVM%20classifier.png)
Fig. 4 - Results with partially occluded objects obtained with STAR detector, SURF extractor, FLANN matcher and SVM classifier



![Fig. 5 - Results obtained with STAR detector, SIFT extractor, FLANN matcher and SVM classifier](https://raw2.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/5%20-%20Results%20obtained%20with%20STAR%20detector,%20SIFT%20extractor,%20FLANN%20matcher%20and%20SVM%20classifier.png)
Fig. 5 - Results obtained with STAR detector, SIFT extractor, FLANN matcher and SVM classifier



![Fig. 6 - Results obtained with SURF detector, SURF extractor, FLANN matcher and ANN classifier](https://raw2.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/6%20-%20Results%20obtained%20with%20SURF%20detector,%20SURF%20extractor,%20FLANN%20matcher%20and%20ANN%20classifier.png)
Fig. 6 - Results obtained with SURF detector, SURF extractor, FLANN matcher and ANN classifier



![Fig. 7 - Results obtained with FAST detector, SURF extractor, FLANN matcher and ANN classifier](https://raw2.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/7%20-%20Results%20obtained%20with%20FAST%20detector,%20SURF%20extractor,%20FLANN%20matcher%20and%20ANN%20classifier.png)
Fig. 7 - Results obtained with FAST detector, SURF extractor, FLANN matcher and ANN classifier



![Fig. 8 - Results obtained with ORB detector, ORB extractor, FLANN matcher and ANN classifier](https://raw2.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/8%20-%20Results%20obtained%20with%20ORB%20detector,%20ORB%20extractor,%20FLANN%20matcher%20and%20ANN%20classifier.png)
Fig. 8 - Results obtained with ORB detector, ORB extractor, FLANN matcher and ANN classifier



## Releases
[Windows 8.1 release](https://github.com/carlosmccosta/Car-Detection/releases)



## Building and developing
The setup instructions on how to build and develop in Visual Studio is available [here](https://github.com/carlosmccosta/Car-Detection/blob/master/CarDetection/docs/Visual%20Studio%20configuration%20for%20OpenCV%202.4.7.txt)
