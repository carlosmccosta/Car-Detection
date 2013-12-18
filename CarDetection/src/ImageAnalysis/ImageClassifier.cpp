#include "ImageClassifier.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifier::ImageClassifier(Ptr<BowVocabulary> bowVocabulary, string classifierFilename, Ptr<StatModel> classifier) :
_bowVocabulary(bowVocabulary), _classifierFilename(classifierFilename), _classifier(classifier) {}
ImageClassifier::~ImageClassifier() {}


bool ImageClassifier::loadClassifier() {
	if (!getBowVocabulary()->loadVocabulary(Mat())) {
		return false;
	}

	stringstream classifierFilenameFull;
	classifierFilenameFull << TRAINING_DIRECTORY << _classifierFilename + CLASSIFIER_EXTENSION;
	FileStorage fs(classifierFilenameFull.str(), FileStorage::READ);
	if (fs.isOpened()) {
		cout << "    -> Loading classifier from " << classifierFilenameFull.str() << endl;
		_classifier->load(classifierFilenameFull.str().c_str(), CLASSIFIER_TAG);
		cout << "    -> Loading finished\n" << endl;
		fs.release();
		return true;
	} else {
		return false;
	}
}


void ImageClassifier::saveClassifier() {
	stringstream classifierFilenameFull;
	classifierFilenameFull << TRAINING_DIRECTORY << _classifierFilename + CLASSIFIER_EXTENSION;
	cout << "    -> Saving classifier to " << classifierFilenameFull.str() << endl;
	_classifier->save(classifierFilenameFull.str().c_str(), CLASSIFIER_TAG);
	cout << "    -> Saving finished\n" << endl;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
