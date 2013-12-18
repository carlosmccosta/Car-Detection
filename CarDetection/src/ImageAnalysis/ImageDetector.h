#pragma once

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>

// project includes
#include "ImageClassifier.h"
#include "../Configs.h"

// namespace specific imports to avoid namespace pollution
using std::string;
using std::vector;

using cv::Mat;
using cv::Ptr;
using cv::Rect;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class DetectorEvaluationResult {
public:
	DetectorEvaluationResult() {}
	DetectorEvaluationResult(double precision, double recall) : _precision(precision), _recall(recall) {}
	virtual ~DetectorEvaluationResult() {}

	// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
	double getPrecision() const { return _precision; }
	void setPrecision(double val) { _precision = val; }
	double getRecall() const { return _recall; }
	void setRecall(double val) { _recall = val; }
	// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

private:
	double _precision;
	double _recall;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ImageDetector {
	public:
		ImageDetector(Ptr<ImageClassifier> imageClassifier);
		virtual ~ImageDetector();

		virtual void detectTargets(Mat& image, vector<Rect>& targetsBoundingRectanglesOut, bool showImageKeyPoints = false) = 0;

		DetectorEvaluationResult evaluateDetector(string testImgsList);

		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		Ptr<ImageClassifier> getImageClassifier() const { return _imageClassifier; }
		void setImageClassifier(Ptr<ImageClassifier> val) { _imageClassifier = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	protected:
		Ptr<ImageClassifier> _imageClassifier;		
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
