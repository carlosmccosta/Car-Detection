#include "ImageDetector.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageDetector::ImageDetector(Ptr<ImageClassifier> imageClassifier) : _imageClassifier(imageClassifier) {}
ImageDetector::~ImageDetector() {}


DetectorEvaluationResult ImageDetector::evaluateDetector(string testImgsList) {
	double precision = 0;
	double recall = 0;

	ifstream imgsList(testImgsList);
	if (imgsList.is_open()) {
		string filename;
		vector<string> fileNames;
		while (getline(imgsList, filename)) {
			stringstream imagePath;
			imagePath << IMGS_DIRECTORY << filename;
			fileNames.push_back(imagePath.str());
		}
		int numberOfFiles = fileNames.size();


		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfFiles; ++i) {
			Mat imagePreprocessed;
			if (_imageClassifier->getBowVocabulary()->getImagePreprocessor()->loadAndPreprocessImage(fileNames[i] + IMAGE_TOKEN, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				vector<Rect> targetsBoundingRectanglesOut;
				detectTargets(imagePreprocessed, targetsBoundingRectanglesOut);


			}
		}

	}

	return DetectorEvaluationResult(precision, recall);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
