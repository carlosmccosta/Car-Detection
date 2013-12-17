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


void ImageUtils::splitKeypoints(string imagePath, const vector<KeyPoint>& keypoints, vector<KeyPoint>& keypointsTargetClass, vector<KeyPoint>& keypointsNonTargetClass) {
	keypointsTargetClass.clear();
	keypointsNonTargetClass.clear();

	vector<Mat> masks;
	loadImageMasks(imagePath, masks);
	int keypointsSize = keypoints.size();

	#pragma omp parallel for schedule(dynamic)
	for (int keyPointPosition = 0; keyPointPosition < keypointsSize; ++keyPointPosition) {
		for (size_t maskPosition = 0; maskPosition < masks.size(); ++maskPosition) {
			Vec3b maskColorInKeyPointPosition = masks[maskPosition].at<Vec3b>(keypoints[keyPointPosition].pt);

			#pragma omp critical
			{
				if (maskColorInKeyPointPosition[2] == 255) {
					keypointsTargetClass.push_back(keypoints[keyPointPosition]);
				}
				else {
					keypointsNonTargetClass.push_back(keypoints[keyPointPosition]);
				}
			}
		}
	}
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
