#include "ImageClassifier.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifier::ImageClassifier(Ptr<BowVocabulary> bowVocabulary, string classifierFilename, Ptr<StatModel> classifier, string classifierName) :
_bowVocabulary(bowVocabulary), _classifierFilename(classifierFilename), _classifier(classifier), _classifierName(classifierName) {}
ImageClassifier::~ImageClassifier() {}



bool ImageClassifier::trainClassifier(const string& vocabularySetupImgsList, const string& classifierTrainImgsList) {
	if (loadClassifier()) {
		return true;
	}

	TrainingData trainingData;
	if (!getBowVocabulary()->computeTrainingData(trainingData, vocabularySetupImgsList, classifierTrainImgsList)) {
		return false;
	}

	Mat& trainingSamples32f = trainingData.getTrainSamples();
	Mat& trainingLabels32s = trainingData.getTrainLabels();
	trainingSamples32f.convertTo(trainingSamples32f, CV_32FC1);
	trainingLabels32s.convertTo(trainingLabels32s, CV_32SC1);

	cout << "\n    -> Training " << _classifierName << " with " << trainingSamples32f.rows << " samples of " << trainingSamples32f.cols << " word size..." << endl;
	PerformanceTimer performanceTimer;
	performanceTimer.start();
	bool trainResult = train(trainingSamples32f, trainingLabels32s);
	cout << "    -> Training finished in " << performanceTimer.getElapsedTimeFormated() << "\n" << endl;	

	saveClassifier();
	return true;
}


float ImageClassifier::analyzeImage(Mat& image, bool drawKeyPoints) {
	int vocabularyWordSize = getBowVocabulary()->getBowImgDescriptorExtractor()->getVocabulary().rows;
	if (vocabularyWordSize < 1 || image.rows < 2 || image.cols < 2) {
		return 0;
	}

	vector<KeyPoint> windowKeyPoints;
	getBowVocabulary()->getFeatureDetector()->detect(image, windowKeyPoints);

	if (windowKeyPoints.size() < 2) {
		return 0.0;
	}

	Mat windowBoWDescriptors;
	getBowVocabulary()->getBowImgDescriptorExtractor()->compute(image, windowKeyPoints, windowBoWDescriptors);
	windowBoWDescriptors.convertTo(windowBoWDescriptors, CV_32FC1);

	if (drawKeyPoints) {
		cv::drawKeypoints(image, windowKeyPoints, image);
	}

	if (windowBoWDescriptors.rows > 0 && windowBoWDescriptors.cols > 0) {
		return predict(windowBoWDescriptors);
	}
	else {
		return 0;
	}
}


bool ImageClassifier::loadClassifier() {
	if (!getBowVocabulary()->loadVocabulary(Mat())) {
		return false;
	}

	stringstream classifierFilenameFull;
	classifierFilenameFull << TRAINING_DIRECTORY << _classifierFilename + CLASSIFIER_EXTENSION;
	FileStorage fs(classifierFilenameFull.str(), FileStorage::READ);
	if (fs.isOpened()) {		
		try {
			_classifier->load(classifierFilenameFull.str().c_str(), CLASSIFIER_TAG);
		} catch(...) {
			cout << "    -> Failed to load classifier from " << classifierFilenameFull.str() << endl;
			fs.release();
			return false;
		}
		cout << "    -> Loaded classifier from " << classifierFilenameFull.str() << endl;
		fs.release();
		return true;
	} else {
		return false;
	}
}


void ImageClassifier::saveClassifier() {
	stringstream classifierFilenameFull;
	classifierFilenameFull << TRAINING_DIRECTORY << _classifierFilename + CLASSIFIER_EXTENSION;	

	try {
		_classifier->save(classifierFilenameFull.str().c_str(), CLASSIFIER_TAG);
		cout << "    -> Saved classifier to " << classifierFilenameFull.str() << endl;
	} catch (...) {
		cout << "    -> Failed to save classifier to " << classifierFilenameFull.str() << endl;
	}		
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageClassifier>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
