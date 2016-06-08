#pragma once

#define _OPENCV_FLANN_HPP_
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/ml/ml.hpp>
#include "clipper.hpp"
#include "../LBP/LBP.h"

#include <memory>
#include <string>
#include <stdio.h>
#include <sys/stat.h>
#include <fstream>

#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <stdlib.h>     // abs

using namespace std;
using namespace ClipperLib;

class Classifier {
    
private:
	cv::Mat1f descriptors;
	cv::Mat1f labels;
	cv::Mat1f responses;	
	cv::SVM svm;
	cv::HOGDescriptor hog;
	cv::SVMParams svmParams;

	LBP lbp;
	
	std::vector<cv::Rect> predictedSlidingWindows;
	std::vector<float> predictedSlidingWindowWeights;

	std::vector<float> classificationPredictions;
	std::vector<float> classificationLabels;

	ostringstream startTime;
	int cnt_Classified;
	int cnt_TP, cnt_TN, cnt_FP, cnt_FN;

	int positiveTrainingWindows, negativeTrainingWindows, discardedTrainingWindows, hardNegativeMinedWindows;
	float overlapThreshold, predictionThreshold;

    	//double calculateOverlap(ClipperLib::Path labelPolygon, ClipperLib::Path slidingWindow);
	ClipperLib::Paths clipPolygon(ClipperLib::Path labelolygon, ClipperLib::Path slidingWindow);
	float calculateOverlapPercentage(ClipperLib::Paths clippedPolygon, ClipperLib::Path slidingWindow);
	float calculateLabel(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage);
	void showTaggedOverlapImage(const cv::Mat& img, ClipperLib::Path labelPolygon, ClipperLib::Path clippedPolygon, cv::Rect slidingWindow, float overlap);
	void shuffleTrainingData(cv::Mat1f  predictionsMatrix, cv::Mat1f labelsMatrix);
public:
    	Classifier();
    	~Classifier();
    	void startTraining();
        void train(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage);
	void trainNegativeSample(const cv::Mat& img, cv::Rect slidingWindow, float imageScaleFactor);
    	void finishTraining();
	void hardNegativeMine(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor);
	void finishHardNegativeMining();
        double classify(const cv::Mat& img, cv::Rect slidingWindow, float imageScaleFactor);
	void evaluate(double prediction, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor);
	void printEvaluation(bool saveResult);
	void showROC(bool saveROC);	
	void generateTaggedResultImage(const cv::Mat& img, string imgName, bool showResult, bool saveResult);
	void evaluateMergedSlidingWindows(const cv::Mat& img, ClipperLib::Path labelPolygon, string imgName, bool showResult, bool saveResult);
	void loadSVM(string path);
	void saveSVM(string path);
};
