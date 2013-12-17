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
		_bowImgDescriptorExtractor->setVocabulary(vocabularyOut);
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

		vector<string> fileNames;
		while (getline(imgsList, filename)) {			
			stringstream imagePath;
			imagePath << IMGS_DIRECTORY << filename << IMAGE_TOKEN;			
			fileNames.push_back(imagePath.str());
		}
		int numberOfFiles = fileNames.size();


		cout << "    -> Building vocabulary with " << numberOfFiles << " images..." << endl;
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
		_bowImgDescriptorExtractor->setVocabulary(vocabularyOut);
		cout << "    -> Building finished\n" << endl;

		saveVocabulary(vocabularyOut);

		_bowTrainer->clear();
		return true;
	}
	
	return false;
}


bool BowVocabulary::computeTrainingData(TrainingData& trainingDataOut, const string& vocabularyImgsList, const string& samplesImgsList) {
	Mat vocabulary;
	if (!computeVocabulary(vocabulary, vocabularyImgsList)) {
		return false;
	}

	ifstream imgsList(vocabularyImgsList);
	if (imgsList.is_open()) {
		string filename;
		vector<string> fileNames;
		while (getline(imgsList, filename)) {
			stringstream imagePath;
			imagePath << IMGS_DIRECTORY << filename;
			fileNames.push_back(imagePath.str());
		}
		int numberOfFiles = fileNames.size();

		int samplesWordSize = _bowImgDescriptorExtractor->getVocabulary().rows;
		Mat trainSamples(0, samplesWordSize, CV_32FC1);
		Mat trainLabels(0, 1, CV_32SC1);

		cout << "    -> Computing " << numberOfFiles << " training samples..." << endl;
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfFiles; ++i) {
			Mat imagePreprocessed;
			if (_imagePreprocessor->loadAndPreprocessImage(fileNames[i] + IMAGE_TOKEN, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				vector<KeyPoint> keypoints;				
				_featureDetector->detect(imagePreprocessed, keypoints);

				vector<KeyPoint> keypointsTargetClass;
				vector<KeyPoint> keypointsNonTargetClass;

				ImageUtils::splitKeypoints(fileNames[i], keypoints, keypointsTargetClass, keypointsNonTargetClass);

				Mat descriptorsTargetClass;
				Mat descriptorsNonTargetClass;

				_bowImgDescriptorExtractor->compute(imagePreprocessed, keypointsTargetClass, descriptorsTargetClass);
				_bowImgDescriptorExtractor->compute(imagePreprocessed, keypointsNonTargetClass, descriptorsNonTargetClass);

				#pragma omp critical
				{
					if (descriptorsTargetClass.rows > 0 && descriptorsTargetClass.cols == samplesWordSize) {
						trainSamples.push_back(descriptorsTargetClass);
						trainLabels.push_back(1);
					}

					if (descriptorsNonTargetClass.rows > 0 && descriptorsNonTargetClass.cols == samplesWordSize) {
						trainSamples.push_back(descriptorsNonTargetClass);
						trainLabels.push_back(0);
					}
				}
			}
		}
		cout << "    -> Training samples finished\n" << endl;

		if (trainSamples.rows != trainLabels.rows || trainSamples.rows == 0 || trainLabels.rows == 0) {
			cout << "\n    !> Invalid training data!\n\n\n" << endl;
			return false;
		}

		trainingDataOut.setTrainSamples(trainSamples);
		trainingDataOut.setTrainLabels(trainLabels);
		return true;
	}

	return false;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<