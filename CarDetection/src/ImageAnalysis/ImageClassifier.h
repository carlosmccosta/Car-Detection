#pragma once

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#define CLASSIFIER_TAG "Classifier"
#define CLASSIFIER_EXTENSION ".xml"
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <string>

// OpenCV includes
#include <opencv2/core/core.hpp>

// project includes
#include "BowVocabulary.h"

// namespace specific imports to avoid namespace pollution
using std::string;

using cv::Mat;
using cv::Ptr;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ClassifierEvaluationResult {
	public:
		ClassifierEvaluationResult(double precision, double recall) : _precision(precision), _recall(recall) {}
		virtual ~ClassifierEvaluationResult() {}

	private:
		double _precision;
		double _recall;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

/// Abstract class for defining Classifiers API
class ImageClassifier {
	public:
		ImageClassifier(Ptr<BowVocabulary> bowVocabulary, string classifierFilename = "classifier.xml");
		virtual ~ImageClassifier();
		
		virtual bool loadClassifier() = 0;
		virtual bool saveClassifier() = 0;
		virtual bool train(const string& vocabularySetupImgsList, const string& classifierTrainImgsList) = 0;
		virtual float predict(Mat& image) = 0;

		ClassifierEvaluationResult evaluateClassifier(string testImgsList);
		
		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		Ptr<BowVocabulary> getBowVocabulary() const { return _bowVocabulary; }
		void setBowVocabulary(Ptr<BowVocabulary> val) { _bowVocabulary = val; }
		string getClassifierFilename() const { return _classifierFilename; }
		void setClassifierFilename(string val) { _classifierFilename = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	protected:
		Ptr<BowVocabulary> _bowVocabulary;
		string _classifierFilename;		
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
