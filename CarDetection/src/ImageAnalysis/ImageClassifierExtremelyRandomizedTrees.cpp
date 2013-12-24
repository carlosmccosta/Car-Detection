#include "ImageClassifierExtremelyRandomizedTrees.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierExtremelyRandomizedTrees::ImageClassifierExtremelyRandomizedTrees(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new ERTrees(), "ERTrees") {}
ImageClassifierExtremelyRandomizedTrees::~ImageClassifierExtremelyRandomizedTrees() {}


bool ImageClassifierExtremelyRandomizedTrees::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {
	return ((Ptr<ERTrees>)_classifier)->train(trainingSamples32f, CV_ROW_SAMPLE, trainingLabels32s);
}


float ImageClassifierExtremelyRandomizedTrees::predict(const Mat& imageBoWDescriptors) {
	return ((Ptr<ERTrees>)_classifier)->predict(imageBoWDescriptors);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
