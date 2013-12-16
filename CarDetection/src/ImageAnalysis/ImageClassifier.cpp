#include "ImageClassifier.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifier::ImageClassifier(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : _bowVocabulary(bowVocabulary), _classifierFilename(classifierFilename) {}
ImageClassifier::~ImageClassifier() {}

ClassifierEvaluationResult ImageClassifier::evaluateClassifier(string testImgsList) {
	double precision = 0;
	double recall = 0;	

	// TODO evaluateClassifier

	return ClassifierEvaluationResult(precision, recall);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
