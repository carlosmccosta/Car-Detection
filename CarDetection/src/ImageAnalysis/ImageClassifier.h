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



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

/// Abstract class for defining Classifiers API
class ImageClassifier {
	public:
		ImageClassifier(Ptr<BowVocabulary> bowVocabulary, string classifierFilename, Ptr<StatModel> classifier, string classifierName);
		virtual ~ImageClassifier();
				
		bool trainClassifier(const string& vocabularySetupImgsList, const string& classifierTrainImgsList);
		virtual bool train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) = 0;

		virtual float analyzeImage(Mat& image, bool drawKeyPoints = false);
		virtual float predict(const Mat& imageBoWDescriptors) = 0;

		bool loadClassifier();
		void saveClassifier();		
		
		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		Ptr<BowVocabulary> getBowVocabulary() { return _bowVocabulary; }
		void setBowVocabulary(Ptr<BowVocabulary> val) { _bowVocabulary = val; }
		string getClassifierFilename() const { return _classifierFilename; }
		void setClassifierFilename(string val) { _classifierFilename = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	protected:
		Ptr<BowVocabulary> _bowVocabulary;
		string _classifierFilename;		
		Ptr<StatModel> _classifier;
		string _classifierName;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
