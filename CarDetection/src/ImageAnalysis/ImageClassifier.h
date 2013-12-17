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
#include <opencv2/ml/ml.hpp>

// project includes
#include "BowVocabulary.h"

// namespace specific imports to avoid namespace pollution
using std::string;

using cv::Mat;
using cv::Ptr;
using cv::StatModel;
using cv::FileStorage;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ClassifierEvaluationResult {
	public:
		ClassifierEvaluationResult() {}
		ClassifierEvaluationResult(double precision, double recall) : _precision(precision), _recall(recall) {}
		virtual ~ClassifierEvaluationResult() {}

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



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

/// Abstract class for defining Classifiers API
class ImageClassifier {
	public:
		ImageClassifier(Ptr<BowVocabulary> bowVocabulary, string classifierFilename, Ptr<StatModel> classifier);
		virtual ~ImageClassifier();
				
		virtual bool train(const string& vocabularySetupImgsList, const string& classifierTrainImgsList) = 0;
		virtual float predict(Mat& image) = 0;

		bool loadClassifier();
		void saveClassifier();
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
		Ptr<StatModel> _classifier;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
