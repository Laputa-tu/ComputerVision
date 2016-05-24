#include "classifier.h"

using namespace std;
using namespace ClipperLib;

cv::Mat1f descriptors;
cv::Mat1f responses;	
cv::SVM svm;
cv::HOGDescriptor hog;
std::vector<cv::Rect> predictedSlidingWindows;
ostringstream startTime;

int cnt_Classified;
int cnt_TP, cnt_TN, cnt_FP, cnt_FN;

/// Constructor
Classifier::Classifier()
{
	cnt_Classified = 0;
	cnt_TP = 0;
	cnt_TN = 0; 
	cnt_FP = 0; 
	cnt_FN = 0;
}

/// Destructor
Classifier::~Classifier() 
{
}

/// Start the training.  This resets/initializes the model.
void Classifier::startTraining()
{
	time_t sTime = time(NULL);
	struct tm *sTimePtr = localtime(&sTime);	
	startTime << sTimePtr->tm_year + 1900 << "_" << sTimePtr->tm_mon + 1 << "_" << sTimePtr->tm_mday << "__" << sTimePtr->tm_hour << "_" << sTimePtr->tm_min << "_" << sTimePtr->tm_sec;	
}

/// Train with a new sliding window section of a training image.
///
/// @param img:  input image
/// @param labelPolygon: a set of points which enwrap the target object
/// @param slidingWindow: the window section of the image that has to be trained
void Classifier::train(const cv::Mat3b& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage)
{	
	//set default values manually
	//ClipperLib::Path labelPolygon;
	//labelPolygon << IntPoint(0, 0) << IntPoint(70, 0) << IntPoint(100, 60) << IntPoint(70, 100) << IntPoint(0, 50);	
	//labelPolygon << IntPoint(0, 0) << IntPoint(1000, 0) << IntPoint(1000, 800) << IntPoint(0, 800);
	//slidingWindow << IntPoint(20, 20) << IntPoint(120, 20) << IntPoint(120, 80) << IntPoint(20, 80);
	//cv::Rect slidingWindow = cv::Rect(0, 0, 64, 128);
	

	//extract slidingWindow out of the image
	cv::Mat3b img2 = img(slidingWindow);

	//calculate Feature-Descriptor
	vector<float> vDescriptor;	
	hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);    
	descriptors.push_back(descriptor);

	
	//calculate Label
	float label = calculateLabel(img, labelPolygon, slidingWindow, imageScaleFactor, showImage);
	responses.push_back(cv::Mat1f(1,1,label));

	if(label == 1.0)
	{
		cv::Rect r = cv::Rect(slidingWindow.x / imageScaleFactor, 
				slidingWindow.y / imageScaleFactor, 
				slidingWindow.width / imageScaleFactor, 
				slidingWindow.height / imageScaleFactor);
		predictedSlidingWindows.push_back ( r );
	}		
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
double Classifier::classify(const cv::Mat3b& img, cv::Rect slidingWindow, float imageScaleFactor)
{		
	//extract slidingWindow out of the image
	cv::Mat3b img2 = img(slidingWindow);

	//calculate Feature-Descriptor
	vector<float> vDescriptor;
	hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);

	//predict Result
	double result = -svm.predict(descriptor, true);
	cout << "Result:              " << result << endl;
	
	if(result > 0) //svm prediction: -1 to +1
	{
		cv::Rect r = cv::Rect(slidingWindow.x / imageScaleFactor, 
				slidingWindow.y / imageScaleFactor, 
				slidingWindow.width / imageScaleFactor, 
				slidingWindow.height / imageScaleFactor);
		predictedSlidingWindows.push_back ( r );	
		result = 1;
	}
	else result = 0;

	return result;
}


void Classifier::evaluate(double prediction, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor)
{	
	//calculate Label
	cv::Mat3b emptyMat;
	float label = calculateLabel(emptyMat, labelPolygon, slidingWindow, imageScaleFactor, false);
	

	if (prediction > 0.5)
	{
		if(label > 0.5) cnt_TP += 1;
		else      	cnt_FP += 1; 
	}
	else
	{
		if(label > 0.5) cnt_FN += 1;
		else     	cnt_TN += 1;
	}
	cnt_Classified += 1;
}


void Classifier::printEvaluation(bool saveResult)
{
	cout << "True Positives:           " << cnt_TP << endl;
	cout << "False Positives:          " << cnt_FP << endl;
	cout << "True Negatives:           " << cnt_TN << endl;
	cout << "False Negatives:          " << cnt_FN << endl;
	cout << "Classified:               " << cnt_Classified << endl;	
	cout << "Accuracy (TP + TN) / All: " << (cnt_TP + cnt_TN) / (1.0 * cnt_Classified) << endl;
	cout << "Recall    TP / (TP + FN): " << cnt_TP / (1.0 * (cnt_TP + cnt_FN)) << endl;
	cout << "Precision TP / (TP + FP): " << cnt_TP / (1.0 * (cnt_TP + cnt_FP)) << endl;
	
	if(saveResult)
	{	
		string dir;
		dir = "./ClassificationResults/";
		mkdir(dir.c_str(), 0777);
		dir = ("./ClassificationResults/" + startTime.str()).c_str();
		mkdir(dir.c_str(), 0777);

		std::ofstream out( (dir + "/" + "_evaluation.txt").c_str() );
		out << "True Positives:           " << cnt_TP << endl;
		out << "False Positives:          " << cnt_FP << endl;
		out << "True Negatives:           " << cnt_TN << endl;
		out << "False Negatives:          " << cnt_FN << endl;
		out << "Classified:               " << cnt_Classified << endl;	
		out << "Accuracy (TP + TN) / All: " << (cnt_TP + cnt_TN) / (1.0 * cnt_Classified) << endl;
		out << "Recall    TP / (TP + FN): " << cnt_TP / (1.0 * (cnt_TP + cnt_FN)) << endl;
		out << "Precision TP / (TP + FP): " << cnt_TP / (1.0 * (cnt_TP + cnt_FP)) << endl;
		out.close();
	}		
}


ClipperLib::Paths Classifier::clipPolygon(ClipperLib::Path labelPolygon, ClipperLib::Path slidingWindow)
{
	ClipperLib::Paths clippedPolygon;

	//perform intersection ...
	Clipper c;
	c.AddPath(labelPolygon, ptSubject, true);
	c.AddPath(slidingWindow, ptClip, true);
	c.Execute(ctIntersection, clippedPolygon, pftNonZero, pftNonZero);

	return clippedPolygon;
}

float Classifier::calculateOverlapPercentage(ClipperLib::Paths clippedPolygon, ClipperLib::Path slidingWindow)
{
	double area_clippedPolygon = 0;
	if (clippedPolygon.size() > 0) area_clippedPolygon = Area(clippedPolygon[0]);
	double area_slidingWindow = Area(slidingWindow);
	double overlap = area_clippedPolygon / area_slidingWindow;

	if(overlap > 0)
	{
		//cout << "Sliding Window:         " << slidingWindow;
		//cout << "Window-Area:          " << area_slidingWindow << endl;
		//cout << "Polygon:                " << labelPolygon;		
		//cout << "Overlap-Area:         " << area_clippedPolygon << endl;
		//cout << "Overlap-Percentage:     " << overlap << endl;
	}
	
	return (float) overlap;
}


float Classifier::calculateLabel(const cv::Mat3b& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage)
{	
	//scale labelPolygon
	int labelPolygonSize = labelPolygon.size();
	for (unsigned i = 0; i < labelPolygonSize; i++)
	{
		labelPolygon[i].X = labelPolygon[i].X * imageScaleFactor;
		labelPolygon[i].Y = labelPolygon[i].Y * imageScaleFactor;
	}	
	
	//calculate intersection/overlap between labelPolygon and slidingWindow
	ClipperLib::Path slidingWindowPath;
	slidingWindowPath << IntPoint(slidingWindow.x, slidingWindow.y)
			<< IntPoint(slidingWindow.x + slidingWindow.width, slidingWindow.y)
			<< IntPoint(slidingWindow.x + slidingWindow.width, slidingWindow.y + slidingWindow.height)
			<< IntPoint(slidingWindow.x, slidingWindow.y + slidingWindow.height); 	
	ClipperLib::Paths clippedPolygon = clipPolygon(labelPolygon, slidingWindowPath);
	float overlapPercentage = calculateOverlapPercentage(clippedPolygon, slidingWindowPath);

	//show tagged Image
	if (clippedPolygon.size() > 0 && overlapPercentage > 0 && showImage) 
	{
		showTaggedOverlapImage(img, labelPolygon, clippedPolygon[0], slidingWindow, overlapPercentage);
		cout << "Clipped Polygon:    " << clippedPolygon[0];
		cout << "Overlap-Percentage: " << overlapPercentage << endl << endl;
	}

	//calculate Label
	float label = 0.0;
	if (overlapPercentage > 0.6) label = 1.0;

	return label;	
}



// generate Image with markers for ALL Sliding Windows that are labeled/classified  
void Classifier::generateTaggedResultImage(const cv::Mat3b& img, string imgName, bool showResult, bool saveResult)
{
	//clone image for drawing shapes
	cv::Mat3b img_show = img.clone(); 

	//draw sliding predicted sliding windows
	for(int i=0; i < predictedSlidingWindows.size(); i++){
	   rectangle( img_show, predictedSlidingWindows[i], cv::Scalar( 0, 255, 0 ), 2, CV_AA, 0 );
	}	
	
	if(showResult)
	{
		cv::imshow("Tagged Image: " + imgName, img_show);	
		cv::waitKey(0);
	}
	if(saveResult)
	{	
		string dir;
		dir = "./ClassificationResults/";
		mkdir(dir.c_str(), 0777);
		dir = ("./ClassificationResults/" + startTime.str()).c_str();
		mkdir(dir.c_str(), 0777);
		cv::imwrite( dir + "/" + imgName, img_show );
	}
		
	// reset sliding window array for next image
	predictedSlidingWindows.clear();
}



// generate Image which shows the clipped polygon and overlap of a single Sliding Window
void Classifier::showTaggedOverlapImage(const cv::Mat3b& img, ClipperLib::Path labelPolygon, ClipperLib::Path clippedPolygon, cv::Rect slidingWindow, float overlap)
{
	//clone image for drawing shapes
	cv::Mat3b img_show = img.clone(); 	


	//draw labelPolygon
	int labelPolygonSize = labelPolygon.size();
	cv::Point lPoly[1][labelPolygonSize];
	for (unsigned i = 0; i < labelPolygonSize; i++)
	{
		lPoly[0][i] = cv::Point( labelPolygon[i].X, labelPolygon[i].Y);
	}		
	const cv::Point* ppt[1] = { lPoly[0] };
	int npt[] = { labelPolygonSize };
	fillPoly( img_show, ppt, npt, 1, cv::Scalar( 255, 255, 255 ), CV_AA );
		

	//draw clipped Polygon
	int clippedPolygonSize = clippedPolygon.size();	
	cv::Point cPoly[1][clippedPolygonSize];
	for (unsigned i = 0; i < clippedPolygonSize; i++)
	{
		cPoly[0][i] = cv::Point( clippedPolygon[i].X, clippedPolygon[i].Y);
	}		
	const cv::Point* ppt2[1] = { cPoly[0] };
	int npt2[] = { clippedPolygonSize };
	fillPoly( img_show, ppt2, npt2, 1, cv::Scalar( 0, 0, 255 ), CV_AA );


	//draw sliding window 
	rectangle( img_show, slidingWindow, cv::Scalar( 0, 255, 0 ), 2, CV_AA, 0 );
	

	//draw Text
	ostringstream s;
	s << "Overlap: " << (floor(overlap * 10000.0) / 100.0) << " %";
	putText(img_show, s.str(), cv::Point(10, 35), cv::FONT_HERSHEY_DUPLEX, 1.3, cv::Scalar( 0, 0, 255 ), 2, CV_AA);

	//Print information to console
	cout << "Sliding Window: " << slidingWindow << endl;
	cout << "Polygon: " << labelPolygon << endl;


	//show image with shapes
	cv::imshow("Overlap Image", img_show);
	cv::waitKey(0);
}


