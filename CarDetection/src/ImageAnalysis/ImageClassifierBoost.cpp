#include "ImageClassifierBoost.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierBoost::ImageClassifierBoost(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new Boost(), "Boost") {}
ImageClassifierBoost::~ImageClassifierBoost() {}


bool ImageClassifierBoost::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {
	return ((Ptr<Boost>)_classifier)->train(trainingSamples32f, CV_ROW_SAMPLE, trainingLabels32s);
}


float ImageClassifierBoost::predict(const Mat& imageBoWDescriptors) {
	return ((Ptr<Boost>)_classifier)->predict(imageBoWDescriptors);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
