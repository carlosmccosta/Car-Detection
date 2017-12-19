# [Car Detection](http://carlosmccosta.github.io/Car-Detection/)


## Overview
This project focus on the detection and recognition of cars in different perspective views and has the following associated paper and presentation:

[Pose Invariant Object Recognition Using a Bag of Words Approach](https://www.researchgate.net/publication/319878114_Pose_Invariant_Object_Recognition_Using_a_Bag_of_Words_Approach)

[Pose Invariant Object Recognition Using a Bag Of Words Approach - Presentation](https://www.researchgate.net/publication/321251682_Pose_Invariant_Object_Recognition_Using_a_Bag_Of_Words_Approach_-_Presentation)


**Abstract:**

Pose invariant object detection and classification plays a crit-ical role in robust image recognition systems and can be applied in amultitude of applications, ranging from simple monitoring to advancedtracking. This paper analyzes the usage of the Bag of Words model forrecognizing objects in different scales, orientations and perspective viewswithin cluttered environments. The recognition system relies on imageanalysis techniques, such as feature detection, description and clusteringalong with machine learning classifiers. For pinpointing the location ofthe target object, it is proposed a multiscale sliding window approach fol-lowed by a dynamic thresholding segmentation. The recognition systemwas tested with several configurations of feature detectors, descriptorsand classifiers and achieved an accuracy of 87% when recognizing carsfrom an annotated dataset with 177 training images and 177 testingimages. 


## Results

![Fig. 1 - Effect of preprocessing (right) in the original image 
(left)](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/1%20-%20Effect%20of%20preprocessing%20(right)%20in%20the%20original%20image%20(left).png)
Fig. 1 - Effect of preprocessing (right) in the original image 
(left)



![Fig. 2 - Target objects ground truth masks](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/2%20-%20Target%20objects%20ground%20truth%20masks.png)
Fig. 2 - Target objects ground truth masks



![Fig. 3 - Results obtained with STAR detector, SIFT extractor, FLANN matcher and ANN classifier](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/3%20-%20Results%20obtained%20with%20STAR%20detector,%20SIFT%20extractor,%20FLANN%20matcher%20and%20ANN%20classifier.png)
Fig. 3 - Results obtained with STAR detector, SIFT extractor, FLANN matcher and ANN classifier



![Fig. 4 - Results with partially occluded objects obtained with STAR detector, SURF extractor, FLANN matcher and SVM classifier](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/4%20-%20Results%20obtained%20with%20STAR%20detector,%20SURF%20extractor,%20FLANN%20matcher%20and%20SVM%20classifier.png)
Fig. 4 - Results with partially occluded objects obtained with STAR detector, SURF extractor, FLANN matcher and SVM classifier



![Fig. 5 - Results obtained with STAR detector, FREAK extractor, FLANN matcher and SVM classifier classifier](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/5%20-%20Results%20obtained%20with%20STAR%20detector,%20FREAK%20extractor,%20FLANN%20matcher%20and%20SVM%20classifier.png)
Fig. 5 - Results obtained with STAR detector, FREAK extractor, FLANN matcher and SVM classifier



![Fig. 6 - Results obtained with STAR detector, SIFT extractor, FLANN matcher and SVM classifier](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/6%20-%20Results%20obtained%20with%20STAR%20detector,%20SIFT%20extractor,%20FLANN%20matcher%20and%20SVM%20classifier.png)
Fig. 6 - Results obtained with STAR detector, SIFT extractor, FLANN matcher and SVM classifier



![Fig. 7 - Results obtained with SURF detector, SURF extractor, FLANN matcher and ANN classifier](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/7%20-%20Results%20obtained%20with%20SURF%20detector,%20SURF%20extractor,%20FLANN%20matcher%20and%20ANN%20classifier.png)
Fig. 7 - Results obtained with SURF detector, SURF extractor, FLANN matcher and ANN classifier



![Fig. 8 - Results obtained with FAST detector, SURF extractor, FLANN matcher and ANN classifier](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/8%20-%20Results%20obtained%20with%20FAST%20detector,%20SURF%20extractor,%20FLANN%20matcher%20and%20ANN%20classifier.png)
Fig. 8 - Results obtained with FAST detector, SURF extractor, FLANN matcher and ANN classifier



![Fig. 9 - Results obtained with ORB detector, ORB extractor, FLANN matcher and ANN classifier](https://raw.github.com/carlosmccosta/Car-Detection/master/Results/Representative%20results/9%20-%20Results%20obtained%20with%20ORB%20detector,%20ORB%20extractor,%20FLANN%20matcher%20and%20ANN%20classifier.png)
Fig. 9 - Results obtained with ORB detector, ORB extractor, FLANN matcher and ANN classifier


## Releases
[Windows 8.1 release](https://github.com/carlosmccosta/Car-Detection/releases)



## Building and developing
The setup instructions on how to build and develop in Visual Studio is available [here](https://github.com/carlosmccosta/Car-Detection/blob/master/CarDetection/docs/Visual%20Studio%20configuration%20for%20OpenCV%202.4.7.txt)
