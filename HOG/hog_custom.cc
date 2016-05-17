#include "hog.h"
#include <opencv2/legacy/legacy.hpp>

#include <math.h>       /* atan2 */

#include <cmath>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;


cv::Mat3b img;

float minv;
float maxv;

float PI;
int channel;
int boxFilterSize;
int blockSize,cellSize,binSize;
int cellCountX,cellCountY,binCount;	
int descriptorSize; 
int descriptorSize2; 
	
Mat1f integralImage;
//Mat1f gradientX;
//Mat1f gradientY;

vector<float> vDescriptor;
vector<float> vDescriptor2;
cv::Mat1f descriptor;
	
	
	
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
/// Start the training.  This resets/initializes the model.
void HOG::startTraining()
{	
	maxv = 0.0;
	minv = 1000.0;

	PI =  3.14159265358979323846;
	channel = 2;
	boxFilterSize = 1;
	
	blockSize = 16;
	cellSize = 8;	
	binSize = 20;
	cellCountX = 8;
	cellCountY = 16;
	binCount = 9;
	
	descriptorSize = cellCountX * cellCountY * binCount; 
	descriptorSize2 = 15 * 7 * 4 * binCount; 
	cout<<"descriptor size: " <<descriptorSize<<endl;
	//int normalizedDescriptorSize = cellCountX * cellCountY * binCount; 	
}


void addToHist(int x, int y, float degrees, float gradientMagnitude)
{
	int cellX = x / cellSize;
	int cellY = y / cellSize;
	int vecParamIndex = cellY*cellCountX*binCount+cellX*binCount;
	
	
	if(degrees <= 10) 
		vDescriptor.at(vecParamIndex + 0) += 1.0;//  * gradientMagnitude;
	else if (degrees >= 170) 
		vDescriptor.at(vecParamIndex + 8) += 1.0;//  * gradientMagnitude;
	else
	{
		//Linear Interpolation
		int binIndex1 = (degrees-10) / 20;
		int binIndex2 = binIndex1 + 1;
		
		float weight2 = (1.0 * ((int)(degrees-10) % 20)) / 20.0;
		float weight1 = 1.0 - weight2;
		
		vDescriptor.at(vecParamIndex + binIndex1) += weight1;// * gradientMagnitude;
		vDescriptor.at(vecParamIndex + binIndex2) += weight2;//  * gradientMagnitude;
		
		
		//if (maxv < weight1)  maxv = weight1 ;
		//if (minv > weight1)  minv = weight1 ;
		
		
		//cout<<"Max: "<<maxv<<endl;
		//cout<<"Min: "<<minv<<endl;
		/*cout<<"x: "<<x<<"    cellX: "<<cellX<<endl;
		cout<<"y: "<<y<<"    cellY: "<<cellY<<endl;
		cout<<"degrees: "<<degrees<<endl;
		cout<<"binIndex1: "<<binIndex1<<endl;
		cout<<"binIndex2: "<<binIndex2<<endl;
		cout<<"weight1: "<<weight1<<endl;
		cout<<"weight2: "<<weight2<<endl;
		cout<<"vecParamIndex: "<<vecParamIndex<<endl;
		cout<<"magnitude: "<<gradientMagnitude<<endl<<endl;*/
	}
	
	
	
	
}

void createDescriptor(const cv::Mat3b& img_default)
{
	img = img_default(cv::Rect((img_default.cols-64)/2,(img_default.rows-128)/2,64,128));	
	integralImage = Mat1f(img.rows, img.cols);
	//gradientX = Mat1f(img.rows / boxFilterSize, img.cols / boxFilterSize);
	//gradientY = Mat1f(img.rows / boxFilterSize, img.cols / boxFilterSize);
	vDescriptor = vector<float>(descriptorSize);
	vDescriptor2 = vector<float>(descriptorSize2);
	
	
	//Create Integral Image		
	/*cout<<"Create Integral Image..."<<endl;
	for (int i = 0; i < img.rows; i++)
	{
		//cout<<i<<endl;
		//fill first column
		if (i == 0) integralImage.at<float>(0,0) = img.at<Vec3b>(0,0)[channel];
		else integralImage.at<float>(i,0) = integralImage.at<float>(i-1, 0) + img.at<Vec3b>(i,0)[channel];
		
		for (int j = 1; j < img.cols; j++)
		{
			//first row
			if (i == 0) integralImage.at<float>(i,j) = integralImage.at<float>(i, j-1) + img.at<Vec3b>(i,j)[channel];
			else integralImage.at<float>(i,j) = integralImage.at<float>(i, j-1) + integralImage.at<float>(i-1, j) + img.at<Vec3b>(i,j)[channel];
		}
	}*/
	//cout<<integralImage<<endl<<endl;
	//imshow("Integral Image", integralImage);
	//waitKey(0);
	
	
	
	//Calculate Gradients
	cout<<"Calculate Gradients..."<<endl;
	float degrees;
	float magnitude;
	float gradientX, gradientY;
	for (int i = boxFilterSize; i < (img.rows - boxFilterSize); i++)
	{
		for (int j = boxFilterSize; j < (img.cols - boxFilterSize); j++)
		{			
			//gradientX.at<float>(i,j) = img.at<Vec3b>(i,j+1)[channel] - img.at<Vec3b>(i,j-1)[channel];
			//gradientY.at<float>(i,j) = img.at<Vec3b>(i+1,j)[channel] - img.at<Vec3b>(i-1,j)[channel];
			
			channel = 0;
			if(img.at<Vec3b>(i,j)[1] > img.at<Vec3b>(i,j)[0]) channel = 1;
			if(img.at<Vec3b>(i,j)[2] > img.at<Vec3b>(i,j)[1]) channel = 2;
			
			gradientX = img.at<Vec3b>(i,j+1)[channel] - img.at<Vec3b>(i,j-1)[channel];
			gradientY = img.at<Vec3b>(i+1,j)[channel] - img.at<Vec3b>(i-1,j)[channel];
			
			degrees = atan2(gradientY, gradientX) * 180.0 / PI; //Range -180 to +180
			if(degrees < 0) degrees = 180.0 + degrees; // Range 0 to +180
			
			magnitude = sqrt(gradientX*gradientX + gradientY*gradientY);
			addToHist(j, i, degrees, magnitude);
			
			//addToHist(126, 62, 169, 2);
		}
	}
	int old;
	cout<<vDescriptor2.size()<<endl;
	for (int i = 0; i < vDescriptor2.size(); i += 36)
	{
		for (int j = 0; j < 9; j++)
		{
			old = i/4 + j + ((int)(i/(4*9*7)))*9;
			//cout<<old<<endl;
			vDescriptor2.at(i + j) = vDescriptor.at(old);
		}
		for (int j = 0; j < 9; j++)
		{
			vDescriptor2.at(i + 9 + j) = vDescriptor.at(i/4 + 9 + j + ((int)(i/(4*9*7)))*9);
		}
		for (int j = 0; j < 9; j++)
		{
			vDescriptor2.at(i + 18 + j) = vDescriptor.at(i/4 + 9*8 + j + ((int)(i/(4*9*7)))*9);
		}
		for (int j = 0; j < 9; j++)
		{
			vDescriptor2.at(i + 27 + j) = vDescriptor.at(i/4 + 9*8 + j + ((int)(i/(4*9*7)))*9);
		}
	}
	
	
	
	//Normalize Histograms
	float vectorLength;
	for (int i = 0; i < vDescriptor2.size(); i += 108)
	{
		vectorLength = 0;
		for (int j = 0; j < 108; j++)
		{
			vectorLength += vDescriptor2.at(i + j) * vDescriptor2.at(i + j);
		}
		vectorLength = sqrt(vectorLength);
		for (int j = 0; j < 108; j++)
		{
			vDescriptor2.at(i + j) /= vectorLength;
		}
	}
	
	
	descriptor = Mat1f(1,vDescriptor2.size(),&vDescriptor2[0]);
	
    //cout<<"Descriptor: "<<descriptor<<endl<<endl;
}
/// Add a new training image.
///
/// @param img:  input image
/// @param bool: value which specifies if img represents a person
void HOG::train(const cv::Mat3b& img_default, bool isPerson)
{
	createDescriptor(img_default);
	
	pimpl->descriptors.push_back(descriptor);
	pimpl->responses.push_back(cv::Mat1f(1,1,float(isPerson)));
	
	
	
	/*
	//cv::Mat3b img = img_default(cv::Rect((img_default.cols-64)/2,(img_default.rows-128)/2,64,128));	
	vector<float> vDescriptor;
	pimpl->hog.compute(img, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);
    cout<<vDescriptor.size()<<endl;
	
	pimpl->descriptors.push_back(descriptor);
	pimpl->responses.push_back(cv::Mat1f(1,1,float(isPerson)));*/
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
double HOG::classify(const cv::Mat3b& img_default)
{
	
	createDescriptor(img_default);
	//cv::Mat3b img2 = img(cv::Rect((img.cols-64)/2,(img.rows-128)/2,64,128));

	//vector<float> vDescriptor;
	//pimpl->hog.compute(img2, vDescriptor);	
	//cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);

	return -pimpl->svm.predict(descriptor, true);
}

