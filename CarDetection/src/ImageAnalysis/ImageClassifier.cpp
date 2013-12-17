#include "ImageClassifier.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifier::ImageClassifier(Ptr<BowVocabulary> bowVocabulary, string classifierFilename, Ptr<StatModel> classifier) :
_bowVocabulary(bowVocabulary), _classifierFilename(classifierFilename), _classifier(classifier) {}
ImageClassifier::~ImageClassifier() {}


bool ImageClassifier::loadClassifier() {
	FileStorage fs(_classifierFilename, FileStorage::READ);
	if (fs.isOpened()) {
		cout << "    -> Loading classifier from " << _classifierFilename << endl;
		_classifier->load(_classifierFilename.c_str(), CLASSIFIER_TAG);
		cout << "    -> Loading finished\n" << endl;
		fs.release();
		return true;
	} else {
		return false;
	}
}


void ImageClassifier::saveClassifier() {
	cout << "    -> Saving classifier to " << _classifierFilename << endl;
	_classifier->save(_classifierFilename.c_str(), CLASSIFIER_TAG);
	cout << "    -> Saving finished\n" << endl;
}


ClassifierEvaluationResult ImageClassifier::evaluateClassifier(string testImgsList) {
	double precision = 0;
	double recall = 0;	

	// TODO evaluateClassifier

	return ClassifierEvaluationResult(precision, recall);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
