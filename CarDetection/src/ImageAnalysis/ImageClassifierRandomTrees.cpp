#include "ImageClassifierRandomTrees.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierRandomTrees::ImageClassifierRandomTrees(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new RandomTrees(), "RandomTrees") {}
ImageClassifierRandomTrees::~ImageClassifierRandomTrees() {}


bool ImageClassifierRandomTrees::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {
	return ((Ptr<RandomTrees>)_classifier)->train(trainingSamples32f, CV_ROW_SAMPLE, trainingLabels32s);
}


float ImageClassifierRandomTrees::predict(const Mat& imageBoWDescriptors) {
	return ((Ptr<RandomTrees>)_classifier)->predict(imageBoWDescriptors);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
