#include "ImageClassifierSVM.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierSVM::ImageClassifierSVM(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new SVM(), "SVM") {}
ImageClassifierSVM::~ImageClassifierSVM() {}


bool ImageClassifierSVM::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {
	Mat varIdx;
	Mat sampleIdx;

	cv::SVMParams params;
	params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, SVM_TRAINING_MAX_ITERATIONS, FLT_EPSILON);
	
	return ((Ptr<SVM>)_classifier)->train_auto(trainingSamples32f, trainingLabels32s, varIdx, sampleIdx, params);
}


float ImageClassifierSVM::predict(const Mat& imageBoWDescriptors) {
	return ((Ptr<SVM>)_classifier)->predict(imageBoWDescriptors, false);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
