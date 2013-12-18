#include "ImageDetectorSlidingWindow.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetectorSlidingWindow>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageDetectorSlidingWindow::ImageDetectorSlidingWindow(Ptr<ImageClassifier> imageClassifier) : ImageDetector(imageClassifier) {}
ImageDetectorSlidingWindow::~ImageDetectorSlidingWindow() {}


Mat ImageDetectorSlidingWindow::detectTargets(Mat& image, vector<Rect>& targetsBoundingRectanglesOut, bool showTargetBoundingRectangles, bool showImageKeyPoints) {
	cout << "    -> Detecting targets..." << endl;
	PerformanceTimer performanceTimer;
	performanceTimer.start();

	float bestMatch = 0;		
	Mat votingMask = Mat::zeros(image.rows, image.cols, CV_8UC1);
	
	int windowMovingXIncrement = (int)(image.cols * SLIDING_WINDOW_MOVING_PERCENTAGE_INCREMENT);
	int windowMovingYIncrement = (int)(image.rows * SLIDING_WINDOW_MOVING_PERCENTAGE_INCREMENT);

	int numberYIterations = image.rows / windowMovingYIncrement;
	int numberOfVotes = 0;

	#pragma omp parallel for schedule(dynamic) reduction(+ : numberOfVotes)
	//for (int yPosition = 0; yPosition <= image.rows; yPosition += windowMovingYIncrement) {
	for (int yIteration = 0; yIteration <= numberYIterations; ++yIteration) {
		int yPosition = yIteration * windowMovingYIncrement;
		for (int xPosition = 0; xPosition < image.cols; xPosition += windowMovingXIncrement) {
			for (double boxSizePercentage = SLIDING_WINDOW_BOX_GROWING_PERCENTAGE_START; boxSizePercentage <= 1.0; boxSizePercentage += SLIDING_WINDOW_BOX_GROWING_PERCENTAGE_INCREMENT) {
				int windowSizeX = (int)(boxSizePercentage * image.cols);
				int windowSizeY = (int)(boxSizePercentage * image.rows);
				if (xPosition + windowSizeX > image.cols || yPosition + windowSizeY > image.rows) {
					break;
				}				
				
				Mat window = image(Rect(xPosition, yPosition, windowSizeX, windowSizeY));
				
				if (window.cols > 6 && window.rows > 6) {
					float prediction = getImageClassifier()->predict(window);
					++numberOfVotes;
					cout << prediction << " ";
					if (prediction > 0.9) {
						for (int xWindow = 0; xWindow < window.cols; ++xWindow) {
							for (int yWindow = 0; yWindow < window.rows; ++yWindow) {
								#pragma omp atomic
								++votingMask.at<unsigned char>(yPosition + yWindow, xPosition + xWindow);
							}
						}
					}
				}
			}
		}
	}
		

	if (showImageKeyPoints) {
		getImageClassifier()->predict(image, showImageKeyPoints);		
	}
	
	Mat detectionMask = votingMask.clone();
	cv::threshold(detectionMask, detectionMask, numberOfVotes * 0.50, 255, CV_THRESH_BINARY);
	ImageUtils::findMaskBoundingRectangles(detectionMask, targetsBoundingRectanglesOut);

	if (showTargetBoundingRectangles) {
		for (size_t i = 0; i < targetsBoundingRectanglesOut.size(); i++) {
			cv::rectangle(image, targetsBoundingRectanglesOut[i], TARGET_DETECTIONS_RECTANGLE_COLOR, 2);
		}
	}

	cout << "    -> Detected " << targetsBoundingRectanglesOut.size() << " targets in " << performanceTimer.getElapsedTimeFormated() << " with " << numberOfVotes << " ROIs" << endl;

	return votingMask;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetectorSlidingWindow>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
