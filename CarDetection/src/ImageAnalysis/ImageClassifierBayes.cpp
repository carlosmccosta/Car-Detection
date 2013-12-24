#include "ImageClassifierBayes.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierBayes::ImageClassifierBayes(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new NormalBayesClassifier(), "NormalBayesClassifier") {}
ImageClassifierBayes::~ImageClassifierBayes() {}


bool ImageClassifierBayes::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {		
	return ((Ptr<NormalBayesClassifier>)_classifier)->train(trainingSamples32f, trainingLabels32s);	
}


float ImageClassifierBayes::predict(const Mat& imageBoWDescriptors) {
	return ((Ptr<NormalBayesClassifier>)_classifier)->predict(imageBoWDescriptors);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
