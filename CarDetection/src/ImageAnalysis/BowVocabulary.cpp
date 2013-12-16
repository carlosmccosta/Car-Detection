#include "BowVocabulary.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
BowVocabulary::BowVocabulary(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher, Ptr<BOWTrainer> bowTrainer,
	Ptr<ImagePreprocessor> imagePreprocessor, string vocabularyFilename) :
	_featureDetector(featureDetector), _descriptorExtractor(descriptorExtractor), _descriptorMatcher(descriptorMatcher), _bowTrainer(bowTrainer),
	_bowImgDescriptorExtractor(new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher)),
	_imagePreprocessor(imagePreprocessor),
	_vocabularyFilename(vocabularyFilename) {}

BowVocabulary::~BowVocabulary() {}


bool BowVocabulary::loadVocabulary(Mat& vocabularyOut) {
	FileStorage fs;

	if (fs.open(_vocabularyFilename, FileStorage::READ)) {
		fs[VOCABULARY_TAG] >> vocabularyOut;
		fs.release();
		return true;
	}
	
	return false;
}


bool BowVocabulary::saveVocabulary(const Mat& vocabularyOut) {
	FileStorage fs;

	if (fs.open(_vocabularyFilename, FileStorage::WRITE)) {
		fs << VOCABULARY_TAG << vocabularyOut;
		fs.release();
		return true;
	}
	
	return false;
}


bool BowVocabulary::computeVocabulary(Mat& vocabularyOut, string vocabularyImgsList) {
	if (loadVocabulary(vocabularyOut)) {
		return true;
	}

	_bowTrainer->clear();

	ifstream imgsList(vocabularyImgsList);
	if (imgsList.is_open()) {
		string filename;

		while (getline(imgsList, filename)) {
			Mat imagePreprocessed;
			if (_imagePreprocessor->loadAndPreprocessImage(filename + IMAGE_TOKEN, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				vector<KeyPoint> keypoints;
				_featureDetector->detect(imagePreprocessed, keypoints);

				Mat descriptors;
				_descriptorExtractor->compute(imagePreprocessed, keypoints, descriptors);

				_bowTrainer->add(descriptors);
			}
		}

		vocabularyOut = _bowTrainer->cluster();
		saveVocabulary(vocabularyOut);

		_bowTrainer->clear();
	}

	_bowTrainer->clear();
	return false;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<