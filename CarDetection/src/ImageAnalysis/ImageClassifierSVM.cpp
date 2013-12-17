#include "ImageClassifierSVM.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierSVM::ImageClassifierSVM(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new SVM()) {}
ImageClassifierSVM::~ImageClassifierSVM() {}


bool ImageClassifierSVM::train(const string& vocabularySetupImgsList, const string& classifierTrainImgsList) {
	if (loadClassifier()) {		
		getBowVocabulary()->loadVocabulary(Mat());
		return true;
	}	
	
	TrainingData trainingData;
	if (!getBowVocabulary()->computeTrainingData(trainingData, vocabularySetupImgsList, classifierTrainImgsList)) {
		return false;
	}

	cv::SVMParams params;
	params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, SVM_TRAINING_MAX_ITERATIONS, FLT_EPSILON);
	
	Mat varIdx;
	Mat samIdx;	

	cout << "    -> Training SVM..." << endl;	
	((Ptr<SVM>)_classifier)->train_auto(trainingData.getTrainSamples(), trainingData.getTrainLabels(), varIdx, samIdx, params);
	cout << "    -> Training finished\n" << endl;

	saveClassifier();
	return true;
}


float ImageClassifierSVM::predict(Mat& image, bool drawKeyPoints) {
	int vocabularyWordSize = getBowVocabulary()->getBowImgDescriptorExtractor()->getVocabulary().rows;
	if (vocabularyWordSize < 1) {
		return 0;
	}

	vector<KeyPoint> windowKeyPoints;
	getBowVocabulary()->getFeatureDetector()->detect(image, windowKeyPoints);

	Mat windowBoWDescriptors(0, vocabularyWordSize, CV_32FC1);
	getBowVocabulary()->getBowImgDescriptorExtractor()->compute(image, windowKeyPoints, windowBoWDescriptors);

	if (drawKeyPoints) {
		cv::drawKeypoints(image, windowKeyPoints, image);
	}

	if (windowBoWDescriptors.rows > 0) {
		return ((Ptr<SVM>)_classifier)->predict(windowBoWDescriptors);
	} else {
		return 0;
	}
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
