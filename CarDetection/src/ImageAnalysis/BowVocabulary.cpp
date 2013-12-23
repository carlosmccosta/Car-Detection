#include "BowVocabulary.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
BowVocabulary::BowVocabulary(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher, Ptr<BOWTrainer> bowTrainer,
	Ptr<ImagePreprocessor> imagePreprocessor, const string& vocabularyFilename, string trainingDataFilename, bool binaryDescriptor) :
	_featureDetector(featureDetector), _descriptorExtractor(descriptorExtractor), _descriptorMatcher(descriptorMatcher), _bowTrainer(bowTrainer),
	_bowImgDescriptorExtractor(new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher)),
	_imagePreprocessor(imagePreprocessor),
	_vocabularyFilename(vocabularyFilename),
	_trainingDataFilename(trainingDataFilename),
	_binaryDescriptor(binaryDescriptor) {}

BowVocabulary::~BowVocabulary() {}


bool BowVocabulary::computeVocabulary(Mat& vocabularyOut, const string& vocabularyImgsList, bool outputAnalyzedImages, bool useOnlyTargetRegions) {
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

		int descriptorsOriginalMatrixType = CV_32FC1;

		//#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfFiles; ++i) {
			Mat imagePreprocessed;
			string imageFilename = IMGS_DIRECTORY + fileNames[i] + IMAGE_TOKEN;
			if (_imagePreprocessor->loadAndPreprocessImage(imageFilename, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				Mat outputImage;
				if (outputAnalyzedImages) {
					outputImage = imagePreprocessed.clone();
				}

				if (useOnlyTargetRegions) {
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
							descriptorsOriginalMatrixType = descriptors.type();
							descriptors.convertTo(descriptors, CV_32FC1);

							if (descriptors.rows > 0) {
								//#pragma omp critical
								_bowTrainer->add(descriptors);
							}

							if (outputAnalyzedImages) {
								cv::drawKeypoints(outputImage, keypoints, outputImage);
							}
						}
					}
				} else {
					vector<KeyPoint> keypoints;
					_featureDetector->detect(imagePreprocessed, keypoints);

					if (keypoints.size() > 3) {
						Mat descriptors;
						_descriptorExtractor->compute(imagePreprocessed, keypoints, descriptors);
						descriptorsOriginalMatrixType = descriptors.type();
						descriptors.convertTo(descriptors, CV_32FC1);

						if (descriptors.rows > 0) {
							//#pragma omp critical
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
		saveVocabulary(vocabularyOut);

		vocabularyOut.convertTo(vocabularyOut, descriptorsOriginalMatrixType);
		_bowImgDescriptorExtractor->setVocabulary(vocabularyOut);		
		cout << "    -> Finished building vocabulary with " << vocabularyOut.rows << " word size in " << performanceTimer.getElapsedTimeFormated() << "\n" << endl;
		

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

	Mat trainSamples;
	Mat trainLabels(0, 1, CV_32SC1);

	if (loadTrainingSamples(trainSamples) && loadTrainingLabels(trainLabels)) {
		trainingDataOut.setTrainSamples(trainSamples);
		trainingDataOut.setTrainLabels(trainLabels);
		return true;
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

		cout << "\n    -> Analysing " << numberOfFiles << " training images..." << endl;
		PerformanceTimer performanceTimer;
		performanceTimer.start();

		//#pragma omp parallel for schedule(dynamic)
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

						//#pragma omp critical
						if (descriptorsTargetClass.rows > 0 && descriptorsTargetClass.cols == samplesWordSize) {
							trainSamples.push_back(descriptorsTargetClass);
							trainLabels.push_back(1);
						}
					}
				}
				
				if (keypointsNonTargetClass.size() > 3) {
					Mat descriptorsNonTargetClass;
					_bowImgDescriptorExtractor->compute(imagePreprocessed, keypointsNonTargetClass, descriptorsNonTargetClass);

					//#pragma omp critical
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

		saveTrainingSamples(trainSamples);
		saveTrainingLabels(trainLabels);
		return true;
	}

	return false;
}


bool BowVocabulary::loadVocabulary(Mat& vocabularyOut) {
	stringstream filename;
	filename << TRAINING_DIRECTORY << _vocabularyFilename << VOCABULARY_EXTENSION;
	if (ImageUtils::loadMatrix(filename.str(), VOCABULARY_TAG, vocabularyOut)) {
		cout << "    -> Loaded vocabulary from " << filename.str() << endl;

		if (_binaryDescriptor) {
			vocabularyOut.convertTo(vocabularyOut, CV_8UC1);
		}

		_bowImgDescriptorExtractor->setVocabulary(vocabularyOut);
		return true;
	}
	return false;
}


bool BowVocabulary::saveVocabulary(const Mat& vocabulary) {
	stringstream filename;
	filename << TRAINING_DIRECTORY << _vocabularyFilename << VOCABULARY_EXTENSION;
	if (ImageUtils::saveMatrix(filename.str(), VOCABULARY_TAG, vocabulary)) {
		cout << "    -> Saved vocabulary to " << filename.str() << endl;
		return true;
	}
	return false;
}


bool BowVocabulary::loadTrainingSamples(Mat& trainingSamplesOut) {
	stringstream filename;
	filename << TRAINING_DIRECTORY << _trainingDataFilename << FILENAME_SEPARATOR << TRAINING_SAMPLES_TAG << TRAINING_SAMPLES_EXTENSION;
	if (ImageUtils::loadMatrix(filename.str(), TRAINING_SAMPLES_TAG, trainingSamplesOut)) {
		cout << "    -> Loaded training samples from " << filename.str() << endl;
		return true;
	}
	return false;
}


bool BowVocabulary::saveTrainingSamples(const Mat& trainingSamples) {
	stringstream filename;
	filename << TRAINING_DIRECTORY << _trainingDataFilename << FILENAME_SEPARATOR << TRAINING_SAMPLES_TAG << TRAINING_SAMPLES_EXTENSION;
	if (ImageUtils::saveMatrix(filename.str(), TRAINING_SAMPLES_TAG, trainingSamples)) {
		cout << "    -> Saved training samples to " << filename.str() << endl;
		return true;
	}
	return false;
}


bool BowVocabulary::loadTrainingLabels(Mat& trainingLabelsOut) {
	stringstream filename;
	filename << TRAINING_DIRECTORY << _trainingDataFilename << FILENAME_SEPARATOR << TRAINING_LABELS_TAG << TRAINING_LABELS_EXTENSION;
	if (ImageUtils::loadMatrix(filename.str(), TRAINING_LABELS_TAG, trainingLabelsOut)) {
		cout << "    -> Loaded training labels from " << filename.str() << endl;
		return true;
	}
	return false;
}


bool BowVocabulary::saveTrainingLabels(const Mat& trainingLabels) {
	stringstream filename;
	filename << TRAINING_DIRECTORY << _trainingDataFilename << FILENAME_SEPARATOR << TRAINING_LABELS_TAG << TRAINING_LABELS_EXTENSION;
	if (ImageUtils::saveMatrix(filename.str(), TRAINING_LABELS_TAG, trainingLabels)) {
		cout << "    -> Saved training labels to " << filename.str() << endl;
		return true;
	}
	return false;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </BowVocabulary>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<