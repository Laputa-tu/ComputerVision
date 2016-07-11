#pragma once

#define _OPENCV_FLANN_HPP_
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/ml/ml.hpp>
#include "clipper.hpp"
#include "../LBP/lbpfeature.cpp"

#include <memory>
#include <string>
#include <stdio.h>
#include <sys/stat.h>
#include <fstream>

#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <stdlib.h>     // abs
#include <math.h>	// min, max

using namespace std;
using namespace ClipperLib;
using namespace cv;

class Classifier {

#define FEATURE_HOG 1
#define FEATURE_LBPH 2
    
private:
	cv::Mat1f descriptors;
	cv::Mat1f labels;
	cv::Mat1f responses;	
	cv::SVM svm;	
	cv::SVMParams svmParams;

    int featureGenerator;
    cv::HOGDescriptor hog;
    LBPFeature lbp;

	ostringstream startTime;
	
	std::vector<cv::Rect> predictedSlidingWindows;
	std::vector<float> predictedSlidingWindowWeights;

	//Sliding-Window-based Evaluation
	std::vector<float> classificationPredictions;
	std::vector<float> classificationLabels;
	float overlapThreshold, predictionThreshold;

	//Merged-Contour-based Evaluation
	std::vector<float> detectionPredictions;
	std::vector<float> detectionLabels;
	float detectionOverlapThreshold;
	std::vector<float> detectionPredictions2;
	std::vector<float> detectionLabels2;
	std::vector<float> detectionPredictions3;
	std::vector<float> detectionLabels3;

	//Counters for Evaluation	
	int cnt_Classified;
	int cnt_TP, cnt_TN, cnt_FP, cnt_FN;
	int positiveTrainingWindows, negativeTrainingWindows, discardedTrainingWindows, hardNegativeMinedWindows;
	

    	//double calculateOverlap(ClipperLib::Path labelPolygon, ClipperLib::Path slidingWindow);
	ClipperLib::Paths clipPolygon(ClipperLib::Path labelolygon, ClipperLib::Path slidingWindow);
	float calculateOverlapPercentage(ClipperLib::Paths clippedPolygon, ClipperLib::Path slidingWindow);
	float calculateLabel(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage);
	void showTaggedOverlapImage(const cv::Mat& img, ClipperLib::Path labelPolygon, ClipperLib::Path clippedPolygon, cv::Rect slidingWindow, float overlap);
	void shuffleTrainingData(cv::Mat1f  predictionsMatrix, cv::Mat1f labelsMatrix);
	cv::Mat cropRotatedRect(const cv::Mat& img, Rect r, float angle);
	cv::vector<Mat> doJitter(Mat img, Rect slidingWindow);
    cv::Mat1f computeFeatureDescriptor(const cv::Mat& img_gray, const cv::Mat& img_color);
public:
    Classifier(float, float, float, int);
    ~Classifier();
    void startTraining(string StartTime);
    void train(const cv::Mat& img_gray, const cv::Mat& img_color, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool doJittering, bool showImage);
    void finishTraining();
    void hardNegativeMine(const cv::Mat& img_gray, const cv::Mat& img_color, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor);
	void finishHardNegativeMining();
    double classify(const cv::Mat& img_gray, const cv::Mat& img_color, cv::Rect slidingWindow, float imageScaleFactor);
	void evaluate(double prediction, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor);
	void printEvaluation(bool saveResult);
	void showROC(bool saveROC);	
	void generateTaggedResultImage(const cv::Mat& img, string imgName, bool showResult, bool saveResult);
	void evaluateMergedSlidingWindows(const cv::Mat& img, ClipperLib::Path labelPolygon, string imgName, bool showResult, bool saveResult);
	void loadSVM(string path);
	void saveSVM(string path);
    void printany();
};
