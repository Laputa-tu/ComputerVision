#pragma once

#define _OPENCV_FLANN_HPP_
#include <opencv/cv.h>
#include <opencv2/ml/ml.hpp>
#include <memory>
#include "clipper.hpp"


class Classifier {
    
private:
	cv::Mat1f descriptors;
	cv::Mat1f responses;	
	cv::SVM svm;
	cv::HOGDescriptor hog;

    	double calculateOverlap(ClipperLib::Path labelPolygon, ClipperLib::Path slidingWindow);

public:
    	Classifier();
    	~Classifier();
    	void startTraining();
    	void train(const cv::Mat3b& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow);
    	void finishTraining();
    	double classify(const cv::Mat3b& img, cv::Rect slidingWindow);

};
