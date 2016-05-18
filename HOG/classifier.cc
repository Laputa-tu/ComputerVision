#include "classifier.h"

using namespace std;
using namespace ClipperLib;

cv::Mat1f descriptors;
cv::Mat1f responses;	
cv::SVM svm;
cv::HOGDescriptor hog;


/// Constructor
Classifier::Classifier()
{
}

/// Destructor
Classifier::~Classifier() 
{
}

double Classifier::calculateOverlap(ClipperLib::Path labelPolygon, ClipperLib::Path slidingWindow)
{
	ClipperLib::Paths clippedPolygon;
	
	//perform intersection ...
	Clipper c;
	c.AddPath(labelPolygon, ptSubject, true);
	c.AddPath(slidingWindow, ptClip, true);
	c.Execute(ctIntersection, clippedPolygon, pftNonZero, pftNonZero);

	double area_clippedPolygon = 0;
	if (clippedPolygon.size() > 0) 
		area_clippedPolygon = Area(clippedPolygon[0]);

	double area_slidingWindow = Area(slidingWindow);
	double overlap = area_clippedPolygon / area_slidingWindow;
	
    if(overlap > 0.05)
    {
        //cout << "Sliding Window:         " << slidingWindow;
        //cout << "Window-Area:          " << area_slidingWindow << endl;
        //cout << "Polygon:                " << labelPolygon;

        //if (clippedPolygon.size() > 0)
            //cout << "Clipped Label-Polygon:  " << clippedPolygon[0];
        //cout << "Overlap-Area:         " << area_clippedPolygon << endl;
        cout << "Overlap-Percentage:     " << overlap << endl << endl;
    }

	
	return overlap;
}




/// Start the training.  This resets/initializes the model.
void Classifier::startTraining()
{
	
}

/// Train with a new sliding window section of a training image.
///
/// @param img:  input image
/// @param labelPolygon: a set of points which enwrap the target object
/// @param slidingWindow: the window section of the image that has to be trained
void Classifier::train(const cv::Mat3b& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor)
{	
	/*
	ClipperLib::Path labelPolygon;
	labelPolygon << IntPoint(0, 0) << IntPoint(70, 0) << IntPoint(100, 60) << IntPoint(70, 100) << IntPoint(0, 50);	
	//slidingWindow << IntPoint(20, 20) << IntPoint(120, 20) << IntPoint(120, 80) << IntPoint(20, 80);
	cv::Rect slidingWindow = cv::Rect(0, 0, 64, 128);
	*/
	
	//scale labelPolygon
	for (unsigned i = 0; i < labelPolygon.size(); i++)
	{
		labelPolygon[i].X = labelPolygon[i].X * imageScaleFactor;
		labelPolygon[i].Y = labelPolygon[i].Y * imageScaleFactor;
	}	

	//extract slidingWindow out of the image
	cv::Mat3b img2 = img(slidingWindow);
	cout << "Sliding Window:         " << slidingWindow << endl;


	//calculate Feature-Descriptor
	vector<float> vDescriptor;	
	hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);    
	descriptors.push_back(descriptor);


	//calculate Label
	ClipperLib::Path slidingWindowPath;
	slidingWindowPath << IntPoint(slidingWindow.x, slidingWindow.y)
			<< IntPoint(slidingWindow.x + slidingWindow.width, slidingWindow.y)
			<< IntPoint(slidingWindow.x + slidingWindow.width, slidingWindow.y + slidingWindow.height)
			<< IntPoint(slidingWindow.x, slidingWindow.y + slidingWindow.height); 
	float label = (float) calculateOverlap(labelPolygon, slidingWindowPath);
    if (label > 0.05) label = 1.0;
    else label = 0.0;
	responses.push_back(cv::Mat1f(1,1,label));
}


/// Finish the training. This finalizes the model. Do not call train() afterwards anymore.
void Classifier::finishTraining()
{
	cv::SVMParams params;
	svm.train( descriptors, responses, cv::Mat(), cv::Mat(), params );
}


/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of having a match inside the sliding window.
///
/// @param img: unknown test image
/// @param slidingWindow: the window section of the image that has to be classified
/// @return: probability of having a match for the target object inside the sliding window section
double Classifier::classify(const cv::Mat3b& img, cv::Rect slidingWindow)
{		
	//cv::Rect slidingWindow = cv::Rect(0, 0, 64, 128);

	//extract slidingWindow out of the image
	cv::Mat3b img2 = img(slidingWindow);
	cout << "Sliding Window:      " << slidingWindow << endl;

	//calculate Feature-Descriptor
	vector<float> vDescriptor;
	hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);

	//predict Result
	double result = -1.0 * svm.predict(descriptor, true);
	cout << "Result:              " << result << endl << endl;
	return result;
}

