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


float ImageClassifierSVM::predict(Mat& image) {
	// TODO predict
	return 0;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
