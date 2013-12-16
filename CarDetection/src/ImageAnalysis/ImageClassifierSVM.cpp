#include "ImageClassifierSVM.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierSVM::ImageClassifierSVM(Ptr<BowVocabulary> bowVocabulary) : ImageClassifier(bowVocabulary) {}
ImageClassifierSVM::~ImageClassifierSVM() {}


bool ImageClassifierSVM::loadClassifier() {
	// TODO loadClassifier
	return false;
}


bool ImageClassifierSVM::saveClassifier() {
	// TODO saveClassifier
	return false;
}


bool ImageClassifierSVM::train(const string& vocabularySetupImgsList, const string& classifierTrainImgsList) {
	// TODO train
	return false;
}


float ImageClassifierSVM::predict(Mat& image) {
	// TODO predict
	return 0;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
