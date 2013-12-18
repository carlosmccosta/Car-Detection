#include "ImageDetector.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageDetector::ImageDetector(Ptr<ImageClassifier> imageClassifier) : _imageClassifier(imageClassifier) {}
ImageDetector::~ImageDetector() {}


DetectorEvaluationResult ImageDetector::evaluateDetector(string testImgsList, bool saveResults) {
	double precision = 0;
	double recall = 0;

	ifstream imgsList(testImgsList);
	if (imgsList.is_open()) {
		string filename;
		vector<string> fileNames;
		while (getline(imgsList, filename)) {
			fileNames.push_back(filename);
		}
		int numberOfFiles = fileNames.size();

		cout << "    -> Evaluating detector..." << endl;
		PerformanceTimer performanceTimer;
		performanceTimer.start();

		//#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfFiles; ++i) {
			Mat imagePreprocessed;
			string imageFilename = IMGS_DIRECTORY + fileNames[i] + IMAGE_TOKEN;

			cout << "\n    -> Evaluating image " << imageFilename << "..." << endl;
			if (_imageClassifier->getBowVocabulary()->getImagePreprocessor()->loadAndPreprocessImage(imageFilename, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				vector<Rect> targetsBoundingRectanglesOut;
				Mat votingMask = detectTargets(imagePreprocessed, targetsBoundingRectanglesOut, true, true);

				if (saveResults) {
					stringstream imageOutputFilename;
					imageOutputFilename << TEST_OUTPUT_DIRECTORY << fileNames[i] << FILENAME_SEPARATOR << _imageClassifier->getClassifierFilename();

					stringstream imageOutputFilenameMask;
					imageOutputFilenameMask << imageOutputFilename.str() << FILENAME_SEPARATOR << DETECTION_VOTE_MASK << IMAGE_OUTPUT_EXTENSION;

					imageOutputFilename << IMAGE_OUTPUT_EXTENSION;

					imwrite(imageOutputFilename.str(), imagePreprocessed);
					imwrite(imageOutputFilenameMask.str(), votingMask);
				}
			}
			cout << "    -> Evaluation of image " << imageFilename << " finished" << endl;
		}
		cout << "\n    -> Finished evaluation of detector in " << performanceTimer.getElapsedTimeFormated() << "\n" << endl;
	}

	return DetectorEvaluationResult(precision, recall);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
