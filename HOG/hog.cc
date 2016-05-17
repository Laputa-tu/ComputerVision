#include "hog.h"

using namespace std;

class HOG::HOGPimpl {
public:

	cv::Mat1f descriptors;
	cv::Mat1f responses;
	
	cv::SVM svm;
	cv::HOGDescriptor hog;
};


/// Constructor
HOG::HOG()
{
	pimpl = std::shared_ptr<HOGPimpl>(new HOGPimpl());
}

/// Destructor
HOG::~HOG() 
{
}


struct Point { 
	int x, y;
};
struct Rect {
	int x, y, width, height;
};



void calculateOverlap(Rect* slidingWindow, Point **labelPolygon) 
{
	cout << "my size (array) " << sizeof(labelPolygon) <<endl;
	cout << "my size (array) " << sizeof(*labelPolygon) <<endl;
	cout << "my size (array) " << sizeof(&labelPolygon) <<endl;

	int polygonSize = sizeof(labelPolygon)/sizeof(labelPolygon[0]);

	cout << "size " << sizeof(labelPolygon) << "  " << sizeof(*labelPolygon) << "\n";

	for(int i = 0; i <= polygonSize; i++)
	{


		//cout << "Point " << i << "   x " << labelPolygon[i].x << "   " << slidingWindow->x << "\n";
	}
}




/// Start the training.  This resets/initializes the model.
void HOG::startTraining()
{
	Rect slidingWindow = {10, 20, 30, 40};

	Point labelPolygon[4];
	labelPolygon[0] = {0, 0};
	labelPolygon[1] = {10, 0};
	labelPolygon[2] = {10, 10};
	labelPolygon[3] = {0, 10};

	cout << sizeof(labelPolygon)<<endl;

	calculateOverlap(&slidingWindow, &labelPolygon);
}

/// Add a new training image.
///
/// @param img:  input image
/// @param float: probability-value which specifies if img represents the class 
void HOG::train(const cv::Mat3b& img, float label)
{
	
	cv::Mat3b img2 = img(cv::Rect((img.cols-64)/2,(img.rows-128)/2,64,128));
	vector<float> vDescriptor;
	pimpl->hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);
    
	pimpl->descriptors.push_back(descriptor);
	pimpl->responses.push_back(cv::Mat1f(1,1,label));
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void HOG::finishTraining()
{
	cv::SVMParams params;
	pimpl->svm.train( pimpl->descriptors, pimpl->responses, cv::Mat(), cv::Mat(), params );
}

/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of being a person.
///
/// @param img: unknown test image
/// @return:    probability of human likelihood
double HOG::classify(const cv::Mat3b& img)
{
	

	cv::Mat3b img2 = img(cv::Rect((img.cols-64)/2,(img.rows-128)/2,64,128));

	vector<float> vDescriptor;
	pimpl->hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);

	return -pimpl->svm.predict(descriptor, true);
}

