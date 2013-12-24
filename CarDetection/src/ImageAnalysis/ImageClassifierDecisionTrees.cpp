#include "ImageClassifierDecisionTrees.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierDecisionTrees::ImageClassifierDecisionTrees(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new DecisionTree(), "DecisionTree") {}
ImageClassifierDecisionTrees::~ImageClassifierDecisionTrees() {}


bool ImageClassifierDecisionTrees::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {	
	return ((Ptr<DecisionTree>)_classifier)->train(trainingSamples32f, CV_ROW_SAMPLE, trainingLabels32s);	
}


float ImageClassifierDecisionTrees::predict(const Mat& imageBoWDescriptors) {
	return (float)((Ptr<DecisionTree>)_classifier)->predict(imageBoWDescriptors)->value;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
