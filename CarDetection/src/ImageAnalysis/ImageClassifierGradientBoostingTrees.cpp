#include "ImageClassifierGradientBoostingTrees.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierGradientBoostingTrees::ImageClassifierGradientBoostingTrees(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new GradientBoostingTrees(), "GradientBoostingTrees") {}
ImageClassifierGradientBoostingTrees::~ImageClassifierGradientBoostingTrees() {}


bool ImageClassifierGradientBoostingTrees::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {
	return ((Ptr<GradientBoostingTrees>)_classifier)->train(trainingSamples32f, CV_ROW_SAMPLE, trainingLabels32s);
}


float ImageClassifierGradientBoostingTrees::predict(const Mat& imageBoWDescriptors) {
	return ((Ptr<GradientBoostingTrees>)_classifier)->predict(imageBoWDescriptors);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
