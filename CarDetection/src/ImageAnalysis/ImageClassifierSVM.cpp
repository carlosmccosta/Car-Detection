#include "ImageClassifierSVM.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierSVM::ImageClassifierSVM(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new SVM()) {}
ImageClassifierSVM::~ImageClassifierSVM() {}


bool ImageClassifierSVM::train(const string& vocabularySetupImgsList, const string& classifierTrainImgsList) {
	if (loadClassifier()) {		
		return true;
	}	
	
	TrainingData trainingData;
	if (!getBowVocabulary()->computeTrainingData(trainingData, vocabularySetupImgsList, classifierTrainImgsList)) {
		return false;
	}
	
	Mat& trainingSamples32f = trainingData.getTrainSamples();	
	Mat& trainingLabels32f = trainingData.getTrainLabels();
	trainingSamples32f.convertTo(trainingSamples32f, CV_32FC1);
	trainingLabels32f.convertTo(trainingLabels32f, CV_32FC1);

	Mat varIdx;
	Mat samIdx;

	cv::SVMParams params;
	params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, SVM_TRAINING_MAX_ITERATIONS, FLT_EPSILON);

	cout << "\n    -> Training SVM with " << trainingSamples32f.rows << " samples of " << trainingSamples32f.cols << " word size with at most " << SVM_TRAINING_MAX_ITERATIONS << " iterations..." << endl;	
	PerformanceTimer performanceTimer;
	performanceTimer.start();
	((Ptr<SVM>)_classifier)->train_auto(trainingSamples32f, trainingLabels32f, varIdx, samIdx, params);
	cout << "    -> Training finished in " << performanceTimer.getElapsedTimeFormated() << "\n" << endl;

	saveClassifier();
	return true;
}


float ImageClassifierSVM::predict(Mat& image, bool drawKeyPoints) {
	int vocabularyWordSize = getBowVocabulary()->getBowImgDescriptorExtractor()->getVocabulary().rows;
	if (vocabularyWordSize < 1 || image.rows < 2 || image.cols < 2) {
		return 0;
	}

	vector<KeyPoint> windowKeyPoints;
	getBowVocabulary()->getFeatureDetector()->detect(image, windowKeyPoints);

	if (windowKeyPoints.size() < 2) {
		return 0.0;
	}

	Mat windowBoWDescriptors;
	getBowVocabulary()->getBowImgDescriptorExtractor()->compute(image, windowKeyPoints, windowBoWDescriptors);
	windowBoWDescriptors.convertTo(windowBoWDescriptors, CV_32FC1);

	if (drawKeyPoints) {
		cv::drawKeypoints(image, windowKeyPoints, image);
	}

	if (windowBoWDescriptors.rows > 0 && windowBoWDescriptors.cols > 0) {
		return ((Ptr<SVM>)_classifier)->predict(windowBoWDescriptors, false);
	} else {
		return 0;
	}
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
