#include "ImageClassifierANN.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageClassifierANN::ImageClassifierANN(Ptr<BowVocabulary> bowVocabulary, string classifierFilename) : ImageClassifier(bowVocabulary, classifierFilename, new NeuralNet_MLP(), "NeuralNet_MLP") {}
ImageClassifierANN::~ImageClassifierANN() {}


bool ImageClassifierANN::train(const Mat& trainingSamples32f, const Mat& trainingLabels32s) {
	Mat layerSizes(1, 3, CV_32SC1);
	layerSizes.at<int>(0) = trainingSamples32f.cols;
	layerSizes.at<int>(1) = 20;
	layerSizes.at<int>(2) = 1;
	((Ptr<NeuralNet_MLP>)_classifier)->create(layerSizes);

	Mat trainingLabels32f;
	trainingLabels32s.convertTo(trainingLabels32f, CV_32FC1);

	Mat sampleWeights = Mat::ones(trainingSamples32f.rows, 1, CV_32FC1);
	return (((Ptr<NeuralNet_MLP>)_classifier)->train(trainingSamples32f, trainingLabels32f, sampleWeights) > 0 ? true : false);
}


float ImageClassifierANN::predict(const Mat& imageBoWDescriptors) {
	Mat outputs;
	((Ptr<NeuralNet_MLP>)_classifier)->predict(imageBoWDescriptors, outputs);

	if (outputs.rows > 0 && outputs.cols > 0 && outputs.type() == CV_32FC1) {
		return outputs.at<float>(0, 0);
	}

	return (float)0;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageClassifierBayes>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
