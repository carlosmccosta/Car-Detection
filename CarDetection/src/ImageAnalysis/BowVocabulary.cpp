#include "BowVocabulary.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
BowVocabulary::BowVocabulary(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher, Ptr<BOWTrainer> bowTrainer,
	Ptr<ImagePreprocessor> imagePreprocessor, const string& vocabularyFilename) :
	_featureDetector(featureDetector), _descriptorExtractor(descriptorExtractor), _descriptorMatcher(descriptorMatcher), _bowTrainer(bowTrainer),
	_bowImgDescriptorExtractor(new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher)),
	_imagePreprocessor(imagePreprocessor),
	_vocabularyFilename(vocabularyFilename) {}

BowVocabulary::~BowVocabulary() {}


bool BowVocabulary::loadVocabulary(Mat& vocabularyOut) {
	FileStorage fs;

	if (fs.open(_vocabularyFilename, FileStorage::READ)) {
		cout << "    -> Loading vocabulary from " << _vocabularyFilename << endl;
		fs[VOCABULARY_TAG] >> vocabularyOut;
		cout << "    -> Loading finished\n" << endl;

		fs.release();
		return true;
	}
	
	return false;
}


bool BowVocabulary::saveVocabulary(const Mat& vocabularyOut) {
	FileStorage fs;

	if (fs.open(_vocabularyFilename, FileStorage::WRITE)) {
		cout << "    -> Saving vocabulary to " << _vocabularyFilename << endl;
		fs << VOCABULARY_TAG << vocabularyOut;
		cout << "    -> Saving finished\n" << endl;

		fs.release();
		return true;
	}
	
	return false;
}


bool BowVocabulary::computeVocabulary(Mat& vocabularyOut, const string& vocabularyImgsList) {
	if (loadVocabulary(vocabularyOut)) {
		return true;
	}	

	_bowTrainer->clear();

	ifstream imgsList(vocabularyImgsList);
	if (imgsList.is_open()) {
		string filename;

		cout << "    -> Building vocabulary..." << endl;

		vector<string> fileNames;
		while (getline(imgsList, filename)) {			
			stringstream imagePath;
			imagePath << IMGS_DIRECTORY << filename << IMAGE_TOKEN;			
			fileNames.push_back(imagePath.str());
		}
		int numberOfFiles = fileNames.size();

		#pragma omp parallel for schedule(dynamic)		
		for (int i = 0; i < numberOfFiles; ++i) {
			Mat imagePreprocessed;
			if (_imagePreprocessor->loadAndPreprocessImage(fileNames[i], imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				vector<KeyPoint> keypoints;
				_featureDetector->detect(imagePreprocessed, keypoints);

				Mat descriptors;
				_descriptorExtractor->compute(imagePreprocessed, keypoints, descriptors);

				#pragma omp critical
				_bowTrainer->add(descriptors);
			}
		}

		vocabularyOut = _bowTrainer->cluster();

		cout << "    -> Building finished\n" << endl;

		saveVocabulary(vocabularyOut);

		_bowTrainer->clear();
	}
	
	return false;
}


bool BowVocabulary::computeTrainingData(TrainingData& trainingDataOut, const string& vocabularyImgsList, const string& samplesImgsList) {
	Mat vocabulary;
	if (!computeVocabulary(vocabulary, vocabularyImgsList)) {
		return false;
	}

	cout << "    -> Computing training samples..." << endl;


	// TODO

	cout << "    -> Training samples finished\n" << endl;


	return true;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<