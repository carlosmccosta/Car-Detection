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

	stringstream vocabularyFilenameFull;
	vocabularyFilenameFull << TRAINING_DIRECTORY << _vocabularyFilename + VOCABULARY_EXTENSION;
	if (fs.open(vocabularyFilenameFull.str(), FileStorage::READ)) {
		cout << "    -> Loading vocabulary from " << vocabularyFilenameFull.str() << endl;
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

	stringstream vocabularyFilenameFull;
	vocabularyFilenameFull << TRAINING_DIRECTORY << _vocabularyFilename + VOCABULARY_EXTENSION;
	if (fs.open(vocabularyFilenameFull.str(), FileStorage::WRITE)) {
		cout << "    -> Saving vocabulary to " << vocabularyFilenameFull.str() << endl;
		fs << VOCABULARY_TAG << vocabularyOut;
		cout << "    -> Saving finished\n" << endl;

		fs.release();
		return true;
	}
	
	return false;
}


bool BowVocabulary::computeVocabulary(Mat& vocabularyOut, const string& vocabularyImgsList, bool outputAnalyzedImages) {
	if (loadVocabulary(vocabularyOut)) {
		return true;
	}	

	_bowTrainer->clear();

	ifstream imgsList(vocabularyImgsList);
	if (imgsList.is_open()) {		
		vector<string> fileNames;
		string filename;
		while (getline(imgsList, filename)) {									
			fileNames.push_back(filename);
		}
		int numberOfFiles = fileNames.size();


		cout << "    -> Building vocabulary with " << numberOfFiles << " images..." << endl;
		PerformanceTimer performanceTimer;
		performanceTimer.start();

		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfFiles; ++i) {
			Mat imagePreprocessed;
			string imageFilename = IMGS_DIRECTORY + fileNames[i] + IMAGE_TOKEN;
			if (_imagePreprocessor->loadAndPreprocessImage(imageFilename, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				Mat outputImage;
				if (outputAnalyzedImages) {
					outputImage = imagePreprocessed.clone();
				}

				vector<Mat> masks;
				ImageUtils::retriveTargetsMasks(IMGS_DIRECTORY + fileNames[i], masks);				
				for (size_t maskIndex = 0; maskIndex < masks.size(); ++maskIndex) {
					vector<KeyPoint> keypoints;
					Mat targetMask = masks[maskIndex];
					_featureDetector->detect(imagePreprocessed, keypoints, targetMask);
					//_featureDetector->detect(imagePreprocessed, keypoints, masks[maskIndex]);

					if (keypoints.size() > 3) {
						Mat descriptors;
						_descriptorExtractor->compute(imagePreprocessed, keypoints, descriptors);
						descriptors.convertTo(descriptors, CV_32FC1);						

						if (descriptors.rows > 0) {
							#pragma omp critical
							_bowTrainer->add(descriptors);
						}

						if (outputAnalyzedImages) {
							cv::drawKeypoints(outputImage, keypoints, outputImage);
						}
					}
				}
				
				if (outputAnalyzedImages) {
					stringstream imageOutputFilename;
					imageOutputFilename << VOCABULARY_BUILD_OUTPUT_DIRECTORY << fileNames[i] << FILENAME_SEPARATOR << _vocabularyFilename << IMAGE_OUTPUT_EXTENSION;
					imwrite(imageOutputFilename.str(), outputImage);
				}				
			}
		}
		vocabularyOut = _bowTrainer->cluster();
		_bowImgDescriptorExtractor->setVocabulary(vocabularyOut);		
		cout << "    -> Finished building vocabulary with " << vocabularyOut.rows << " word size in " << performanceTimer.getElapsedTimeFormated() << "\n" << endl;

		saveVocabulary(vocabularyOut);

		_bowTrainer->clear();
		return true;
	}
	
	return false;
}


bool BowVocabulary::computeTrainingData(TrainingData& trainingDataOut, const string& vocabularyImgsList, const string& samplesImgsList, bool outputAnalyzedImages) {	
	if (_bowImgDescriptorExtractor->getVocabulary().rows == 0) {
		Mat vocabulary;
		if (!computeVocabulary(vocabulary, vocabularyImgsList)) {
			return false;
		}
	}

	ifstream imgsList(vocabularyImgsList);
	if (imgsList.is_open()) {		
		vector<string> fileNames;
		string filename;
		while (getline(imgsList, filename)) {			
			fileNames.push_back(filename);
		}
		int numberOfFiles = fileNames.size();

		int samplesWordSize = _bowImgDescriptorExtractor->getVocabulary().rows;
		Mat trainSamples;
		Mat trainLabels;

		cout << "    -> Analysing " << numberOfFiles << " training images..." << endl;
		PerformanceTimer performanceTimer;
		performanceTimer.start();

		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfFiles; ++i) {
			Mat imagePreprocessed;
			string imageFilenameShort = IMGS_DIRECTORY + fileNames[i];
			string imageFilenameFull = imageFilenameShort + IMAGE_TOKEN;
			if (_imagePreprocessor->loadAndPreprocessImage(imageFilenameFull, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				vector<KeyPoint> keypoints;				
				_featureDetector->detect(imagePreprocessed, keypoints);

				vector< vector <KeyPoint> > keypointsTargetClass;
				vector<KeyPoint> keypointsNonTargetClass;

				ImageUtils::splitKeyPoints(imageFilenameShort, keypoints, keypointsTargetClass, keypointsNonTargetClass);

				for (size_t targetClassInstancePosition = 0; targetClassInstancePosition < keypointsTargetClass.size(); ++targetClassInstancePosition) {
					if (keypointsTargetClass[targetClassInstancePosition].size() > 3) {
						Mat descriptorsTargetClass;						
						_bowImgDescriptorExtractor->compute(imagePreprocessed, keypointsTargetClass[targetClassInstancePosition], descriptorsTargetClass);

						#pragma omp critical
						if (descriptorsTargetClass.rows > 0 && descriptorsTargetClass.cols == samplesWordSize) {
							trainSamples.push_back(descriptorsTargetClass);
							trainLabels.push_back(1);
						}
					}
				}
				
				if (keypointsNonTargetClass.size() > 3) {
					Mat descriptorsNonTargetClass;
					_bowImgDescriptorExtractor->compute(imagePreprocessed, keypointsNonTargetClass, descriptorsNonTargetClass);

					#pragma omp critical
					if (descriptorsNonTargetClass.rows > 0 && descriptorsNonTargetClass.cols == samplesWordSize) {
						trainSamples.push_back(descriptorsNonTargetClass);
						trainLabels.push_back(0);
					}
				}

				if (outputAnalyzedImages) {
					stringstream imageOutputFilename;
					imageOutputFilename << SAMPLES_BUILD_OUTPUT_DIRECTORY << fileNames[i] << FILENAME_SEPARATOR << _vocabularyFilename << IMAGE_OUTPUT_EXTENSION;
					for (size_t targetClassInstancePosition = 0; targetClassInstancePosition < keypointsTargetClass.size(); ++targetClassInstancePosition) {
						if (keypointsTargetClass[targetClassInstancePosition].size() > 0) {
							cv::drawKeypoints(imagePreprocessed, keypointsTargetClass[targetClassInstancePosition], imagePreprocessed, TARGET_KEYPOINT_COLOR);
						}
					}

					if (keypointsNonTargetClass.size() > 0) {
						cv::drawKeypoints(imagePreprocessed, keypointsNonTargetClass, imagePreprocessed, NONTARGET_KEYPOINT_COLOR);
					}

					imwrite(imageOutputFilename.str(), imagePreprocessed);
				}					
			}
		}
		cout << "    -> Computed " << trainSamples.rows << " training samples from " << numberOfFiles << " images in " << performanceTimer.getElapsedTimeFormated() << "\n" << endl;

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