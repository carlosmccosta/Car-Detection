#pragma once

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class DetectorEvaluationResult {
public:
	DetectorEvaluationResult();
	DetectorEvaluationResult(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives);
	DetectorEvaluationResult(double precision, double recall, double accuracy);
	virtual ~DetectorEvaluationResult() {}

	static double computePrecision(size_t truePositives, size_t falsePositives);
	static double computeRecall(size_t truePositives, size_t falseNegatives);
	static double computeAccuracy(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives);

	// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
	double getPrecision() const { return _precision; }
	void setPrecision(double val) { _precision = val; }
	double getRecall() const { return _recall; }
	void setRecall(double val) { _recall = val; }
	double getAccuracy() const { return _accuracy; }
	void setAccuracy(double val) { _accuracy = val; }

	size_t getTruePositives() const { return _truePositives; }
	void setTruePositives(size_t val) { _truePositives = val; }
	size_t getTrueNegatives() const { return _trueNegatives; }
	void setTrueNegatives(size_t val) { _trueNegatives = val; }
	size_t getFalsePositives() const { return _falsePositives; }
	void setFalsePositives(size_t val) { _falsePositives = val; }
	size_t getFalseNegatives() const { return _falseNegatives; }
	void setFalseNegatives(size_t val) { _falseNegatives = val; }
	// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

private:
	size_t _truePositives = 0;	
	size_t _trueNegatives = 0;	
	size_t _falsePositives = 0;	
	size_t _falseNegatives = 0;	

	double _precision;	// truePositives / (truePositives + falsePositives)
	double _recall;		// truePositives / (truePositives + falseNegatives)
	double _accuracy;	// (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)	
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

