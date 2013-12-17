#include "ImageUtils.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void ImageUtils::loadImageMasks(string imagePath, vector<Mat>& masks) {
	size_t imageMaskNumber = 0;
	bool masksAvailable = true;

	while (masksAvailable) {
		stringstream imageMaskPath;
		imageMaskPath << imagePath << MASK_TOKEN << imageMaskNumber << MASK_EXTENSION;
		
		Mat mask = imread(imageMaskPath.str(), CV_LOAD_IMAGE_COLOR);
		if (mask.data) {
			masks.push_back(mask);
		} else {
			masksAvailable = false;
		}

		++imageMaskNumber;
	}	
}


void ImageUtils::splitKeyPoints(string imagePath, const vector<KeyPoint>& keypoints, vector< vector <KeyPoint> >& keypointsTargetClass, vector<KeyPoint>& keypointsNonTargetClass) {
	keypointsTargetClass.clear();
	keypointsNonTargetClass.clear();

	vector<Mat> masks;
	loadImageMasks(imagePath, masks);
	int keyPointsSize = keypoints.size();
	vector<bool> targetKeypoints(keypoints.size());

	keypointsTargetClass.resize(masks.size());

	#pragma omp parallel for schedule(dynamic)
	for (int keyPointPosition = 0; keyPointPosition < keyPointsSize; ++keyPointPosition) {
		bool keyPointIsNotCar = true;

		for (size_t maskPosition = 0; maskPosition < masks.size(); ++maskPosition) {									
			Vec3b maskColorInKeyPointPosition = masks[maskPosition].at<Vec3b>(keypoints[keyPointPosition].pt);
						
			if (maskColorInKeyPointPosition[2] == 255) {
				#pragma omp critical
				keypointsTargetClass[maskPosition].push_back(keypoints[keyPointPosition]);
				
				keyPointIsNotCar = false;
				break;
			}			
		}
		
		if (keyPointIsNotCar) {
			#pragma omp critical
			keypointsNonTargetClass.push_back(keypoints[keyPointPosition]);
		}
	}
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
