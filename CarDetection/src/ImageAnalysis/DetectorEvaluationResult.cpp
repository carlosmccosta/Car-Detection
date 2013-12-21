#include "DetectorEvaluationResult.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
DetectorEvaluationResult::DetectorEvaluationResult() {}
DetectorEvaluationResult::DetectorEvaluationResult(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives) :
	_truePositives(truePositives), _trueNegatives(trueNegatives), _falsePositives(falsePositives), _falseNegatives(falseNegatives) {
	
	_precision = computePrecision(truePositives, falsePositives);
	_recall = computeRecall(truePositives, falseNegatives);
	_accuracy = computeAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives);
}


DetectorEvaluationResult::DetectorEvaluationResult(double precision, double recall, double accuracy) :
	_precision(precision), _recall(recall), _accuracy(accuracy) {}


double DetectorEvaluationResult::computePrecision(size_t truePositives, size_t falsePositives) {
	double divisor = truePositives + falsePositives;
	if (divisor == 0) {
		return 0;
	}

	return (double)truePositives / divisor;
}


double DetectorEvaluationResult::computeRecall(size_t truePositives, size_t falseNegatives) {
	double divisor = truePositives + falseNegatives;
	if (divisor == 0) {
		return 0;
	}

	return (double)truePositives / divisor;
}


double DetectorEvaluationResult::computeAccuracy(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives) {
	double divisor = truePositives + trueNegatives + falsePositives + falseNegatives;
	if (divisor == 0) {
		return 0;
	}

	return (double)(truePositives + trueNegatives) / divisor;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<