#include "ImageClassifierSVM.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierSVM::ImageClassifierSVM(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename) {}
ImageClassifierSVM::~ImageClassifierSVM() {}


bool ImageClassifierSVM::loadClassifier() {
	_classifier.load(getClassifierFilename().c_str());
	return (_classifier.get_support_vector_count() != 0);	
}


void ImageClassifierSVM::saveClassifier() {
	_classifier.save(getClassifierFilename().c_str());	
}


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
	Mat trainSamples32f;
	Mat trainLabels32f;

	trainingData.getTrainSamples().convertTo(trainSamples32f, CV_32F);
	trainingData.getTrainLabels().convertTo(trainLabels32f, CV_32F);

	cout << "    -> Training SVM..." << endl;
	_classifier.train_auto(trainSamples32f, trainLabels32f, varIdx, samIdx, params);	
	cout << "    -> Training finished\n" << endl;

	saveClassifier();
	return true;
}


float ImageClassifierSVM::predict(Mat& image) {
	// TODO predict
	return 0;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
