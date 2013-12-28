#include "ImageClassifierKNN.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierKNN::ImageClassifierKNN(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new KNearest(), "KNearest") {}
ImageClassifierKNN::~ImageClassifierKNN() {}


bool ImageClassifierKNN::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {		
	return ((Ptr<KNearest>)_classifier)->train(trainingSamples32f, trainingLabels32s);	
}


float ImageClassifierKNN::predict(const Mat& imageBoWDescriptors) {
	return ((Ptr<KNearest>)_classifier)->find_nearest(imageBoWDescriptors, KNN_CLASSIFIER_K);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifierSVM>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
