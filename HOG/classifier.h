#pragma once

#define _OPENCV_FLANN_HPP_
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/ml/ml.hpp>
#include "clipper.hpp"

#include <memory>
#include <string>
#include <stdio.h>
#include <sys/stat.h>
#include <fstream>

using namespace std;
using namespace ClipperLib;

class Classifier {
    
private:
	cv::Mat1f descriptors;
	cv::Mat1f responses;	
	cv::SVM svm;
	cv::HOGDescriptor hog;
	std::vector<cv::Rect> predictedSlidingWindows;

    	//double calculateOverlap(ClipperLib::Path labelPolygon, ClipperLib::Path slidingWindow);
	ClipperLib::Paths clipPolygon(ClipperLib::Path labelolygon, ClipperLib::Path slidingWindow);
	float calculateOverlapPercentage(ClipperLib::Paths clippedPolygon, ClipperLib::Path slidingWindow);
	float calculateLabel(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage);
	void showTaggedOverlapImage(const cv::Mat& img, ClipperLib::Path labelPolygon, ClipperLib::Path clippedPolygon, cv::Rect slidingWindow, float overlap);
public:
    	Classifier();
    	~Classifier();
    	void startTraining();
        void train(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage);
    	void finishTraining();
        double classify(const cv::Mat& img, cv::Rect slidingWindow, float imageScaleFactor);
	void evaluate(double prediction, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor);
	void printEvaluation(bool saveResult);
	void showROC(bool saveROC);
	
	void generateTaggedResultImage(const cv::Mat& img, string imgName, bool showResult, bool saveResult);
};
