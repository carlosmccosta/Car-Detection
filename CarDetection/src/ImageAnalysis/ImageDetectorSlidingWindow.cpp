#include "ImageDetectorSlidingWindow.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetectorSlidingWindow>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageDetectorSlidingWindow::ImageDetectorSlidingWindow(Ptr<ImageClassifier> imageClassifier) : ImageDetector(imageClassifier) {}
ImageDetectorSlidingWindow::~ImageDetectorSlidingWindow() {}


void ImageDetectorSlidingWindow::detectTargets(Mat& image, vector<Rect>& targetsBoundingRectanglesOut, bool showImageKeyPoints) {
	cout << "    -> Detecting targets..." << endl;
	
	float bestMatch = 0;		
	Mat votingMask = Mat::zeros(image.rows, image.cols, CV_8UC1);
	
	int windowMovingXIncrement = (int)(image.cols * SLIDING_WINDOW_MOVING_PERCENTAGE_INCREMENT);
	int windowMovingYIncrement = (int)(image.rows * SLIDING_WINDOW_MOVING_PERCENTAGE_INCREMENT);

	//#pragma omp parallel for schedule(dynamic)	
	for (int yPosition = 0; yPosition <= image.rows; yPosition += windowMovingYIncrement) {
		for (int xPosition = 0; xPosition < image.cols; xPosition += windowMovingXIncrement) {
			for (double boxSizePercentage = SLIDING_WINDOW_BOX_GROWING_PERCENTAGE_INCREMENT; boxSizePercentage <= 1.0; boxSizePercentage += SLIDING_WINDOW_BOX_GROWING_PERCENTAGE_INCREMENT) {
				Mat window = image(Rect(xPosition, yPosition,
					std::min((int)(boxSizePercentage * image.cols), image.cols - xPosition),
					std::min((int)(boxSizePercentage * image.rows), image.rows - yPosition)));
				
				float prediction = getImageClassifier()->predict(window);
				if (prediction == 1.0) {
					for (int xWindow = 0; xWindow < window.cols; ++xWindow) {
						for (int yWindow = 0; yWindow < window.rows; ++yWindow) {
							//#pragma omp atomic
							++votingMask.at<unsigned char>(xPosition + xWindow, yPosition + yWindow);
						}
					}
				}
			}
		}
	}
	
	if (showImageKeyPoints) {
		getImageClassifier()->predict(image, showImageKeyPoints);
	}
	
	ImageUtils::findMaskBoundingRectangles(votingMask, targetsBoundingRectanglesOut);	

	cout << "    -> Detection finished\n" << endl;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetectorSlidingWindow>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
