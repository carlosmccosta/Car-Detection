#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#define VOCABULARY_TAG "Vocabulary"
#define VOCABULARY_EXTENSION ".xml"
#define TRAINING_SAMPLES_TAG "trainingSamples"
#define TRAINING_SAMPLES_EXTENSION ".xml"
#define TRAINING_LABELS_TAG "trainingLabels"
#define TRAINING_LABELS_EXTENSION ".xml"
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "ImagePreprocessor.h"
#include "ImageUtils.h"
#include "../libs/PerformanceTimer.h"
#include "../Configs.h"


// namespace specific imports to avoid namespace pollution
using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::stringstream;
using std::ifstream;
using std::getline;

using cv::Mat;
using cv::Rect;
using cv::RotatedRect;
using cv::Ptr;
using cv::KeyPoint;
using cv::Scalar;
using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::BOWTrainer;
using cv::BOWImgDescriptorExtractor;
using cv::FileStorage;
using cv::imwrite;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Training data>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class TrainingData {
public:
	TrainingData() {}
	TrainingData(const Mat& trainSamples, const Mat& trainLabels) : _trainSamples(trainSamples), _trainLabels(trainLabels) {}
	virtual ~TrainingData() {}

	// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
	Mat& getTrainSamples() { return _trainSamples; }
	void setTrainSamples(const Mat& val) { _trainSamples = val; }
	Mat& getTrainLabels() { return _trainLabels; }
	void setTrainLabels(const Mat& val) { _trainLabels = val; }
	// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

private:
	Mat _trainSamples;	
	Mat _trainLabels;	
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Training data>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class BowVocabulary {
	public:
		BowVocabulary(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher, Ptr<BOWTrainer> bowTrainer,
			Ptr<ImagePreprocessor> imagePreprocessor, const string& vocabularyFilename, string trainingDataFilename, bool binaryDescriptor);
		virtual ~BowVocabulary();

		bool computeVocabulary(Mat& vocabularyOut, const string& vocabularyImgsList, bool outputAnalyzedImages = true, bool useOnlyTargetRegions = true);
		bool computeTrainingData(TrainingData& trainingDataOut, const string& vocabularyImgsList, const string& samplesImgsList, bool outputAnalyzedImages = true);

		bool loadVocabulary(Mat& vocabularyOut);
		bool saveVocabulary(const Mat& vocabulary);
		bool loadTrainingSamples(Mat& trainingSamplesOut);
		bool saveTrainingSamples(const Mat& trainingSamples);
		bool loadTrainingLabels(Mat& trainingLabelsOut);
		bool saveTrainingLabels(const Mat& trainingLabels);


		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		Ptr<FeatureDetector> getFeatureDetector() const { return _featureDetector; }
		void setFeatureDetector(Ptr<FeatureDetector> val) { _featureDetector = val; }
		Ptr<DescriptorExtractor> getDescriptorExtractor() const { return _descriptorExtractor; }
		void setDescriptorExtractor(Ptr<DescriptorExtractor> val) { _descriptorExtractor = val; }
		Ptr<DescriptorMatcher> getDescriptorMatcher() const { return _descriptorMatcher; }
		void setDescriptorMatcher(Ptr<DescriptorMatcher> val) { _descriptorMatcher = val; }

		Ptr<BOWTrainer> getBowTrainer() const { return _bowTrainer; }
		void setBowTrainer(Ptr<BOWTrainer> val) { _bowTrainer = val; }
		Ptr<BOWImgDescriptorExtractor> getBowImgDescriptorExtractor() { return _bowImgDescriptorExtractor; }
		void setBowImgDescriptorExtractor(Ptr<BOWImgDescriptorExtractor> val) { _bowImgDescriptorExtractor = val; }


		Ptr<ImagePreprocessor> getImagePreprocessor() const { return _imagePreprocessor; }
		void setImagePreprocessor(Ptr<ImagePreprocessor> val) { _imagePreprocessor = val; }

		string getVocabularyFilename() const { return _vocabularyFilename; }
		void setVocabularyFilename(const string& val) { _vocabularyFilename = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	private:
		Ptr<FeatureDetector> _featureDetector;		
		Ptr<DescriptorExtractor> _descriptorExtractor;		
		Ptr<DescriptorMatcher> _descriptorMatcher;
		
		Ptr<BOWTrainer> _bowTrainer;				
		Ptr<BOWImgDescriptorExtractor> _bowImgDescriptorExtractor;
		
		Ptr<ImagePreprocessor> _imagePreprocessor;
		
		string _vocabularyFilename;
		string _trainingDataFilename;

		bool _binaryDescriptor;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
