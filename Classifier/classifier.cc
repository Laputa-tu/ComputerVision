#include "classifier.h"

using namespace std;
using namespace ClipperLib;

/// Constructor
Classifier::Classifier(float overlapLabels, float predictionThresh, float overlapDetector)
{
    overlapThreshold = overlapLabels;		// label = Percentage of overlap -> 0 to 1.0
    predictionThreshold = predictionThresh;	// svm prediction: -1 to +1
    detectionOverlapThreshold = overlapDetector;	// overlap of the merged-slidingWindow-contour and the labelPolygon

	cnt_Classified = 0;
	cnt_TP = 0;
	cnt_TN = 0; 
	cnt_FP = 0; 
	cnt_FN = 0;	

	positiveTrainingWindows = 0;
	negativeTrainingWindows = 0;
	discardedTrainingWindows = 0;	
	hardNegativeMinedWindows = 0;

	svmParams.svm_type    = CvSVM::C_SVC;
	svmParams.kernel_type = CvSVM::LINEAR;
	svmParams.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// initialize random seed   
	unsigned int t = time(NULL);
	srand (t);
	cout << "srand: " << t << endl;
}

/// Destructor
Classifier::~Classifier() 
{
}

/// Start the training.  This resets/initializes the model.
void Classifier::startTraining(string StartTime)
{
    startTime << StartTime;
}

vector<Mat> Classifier::doJitter(Mat img, Rect slidingWindow)
{
    vector<Mat> additionalImages;
    Rect slidingWindow_jitter;
    Mat img_crop;
    int randomX, randomY;
    int jitterMinX, jitterMaxX, jitterMinY, jitterMaxY;
    int numberOfJitters = 3;
    int jitterX=8, jitterY=8;
    int startAngle = -24, endAngle = 24, stepAngle = 8;

    // use no jitter in the first loop
    randomX = 0;
    randomY = 0;    

    for (int i = 0; i < numberOfJitters; i++)
    {
        slidingWindow_jitter = Rect(slidingWindow.x + randomX, slidingWindow.y + randomY, slidingWindow.width, slidingWindow.height);
        for(int angle = startAngle; angle <= endAngle; angle += stepAngle)
        {
            img_crop = cropRotatedRect(img, slidingWindow_jitter, angle);
            if(!img_crop.empty())
            {
                additionalImages.push_back(img_crop);
                cv::flip(img_crop, img_crop, 1);
                additionalImages.push_back(img_crop);
            }
        }
	
	// calculate random jitter for next loop
	jitterMinX = max(-jitterX, -slidingWindow.x);
        jitterMinY = max(-jitterY, -slidingWindow.y);
        jitterMaxX = min(jitterX, img.cols - slidingWindow.x - slidingWindow.width);
        jitterMaxY = min(jitterY, img.rows - slidingWindow.y - slidingWindow.height);
        randomX = rand() % (jitterMaxX - jitterMinX + 1) + jitterMinX;
        randomY = rand() % (jitterMaxY - jitterMinY + 1) + jitterMinY;	
    }

    return additionalImages;
}

cv::Mat Classifier::cropRotatedRect(const cv::Mat& img, Rect rect, float angle)
{
    Mat M, rotatedImage, nullImage;
    Size rect_size = rect.size();
    Point rect_center(rect.x + 0.5 * rect.width, rect.y + 0.5 * rect.height);

    // create Rotated Rectangle
    RotatedRect rotRect(rect_center, rect_size, angle);


    // check if rotated rectangle is outside of the image bounds
    bool rectangleOutOfBounds = false;
    Point2f vertices[4];
    rotRect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        if(vertices[i].x < 0 || vertices[i].x >= img.cols || vertices[i].y < 0 || vertices[i].y >= img.rows)
        {
            //Rotated rect is out of image bounds
            return nullImage;
        }
    }

    if (angle < -45.)
    {
        angle += 90.0;
        swap(rect_size.width, rect_size.height);
    }
    M = getRotationMatrix2D(rect_center, angle, 1.0);
    warpAffine(img, rotatedImage, M, img.size(), INTER_CUBIC);

    return rotatedImage(rect);
}

/// Train with a new sliding window section of a training image.
///
/// @param img:  input image
/// @param labelPolygon: a set of points which enwrap the target object
/// @param slidingWindow: the window section of the image that has to be trained
void Classifier::train(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage)
{		
    vector<Mat> additionalImages;

    //extract slidingWindow and convert to grayscale
    cv::Mat img2 = img(slidingWindow);
    cvtColor(img2,img2,CV_RGB2GRAY);

	//calculate Feature-Descriptor
	vector<float> vDescriptor;	
    transpose(img2, img2);
	hog.compute(img2, vDescriptor);
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);
	//lbp.compute(img2, vDescriptor);

	//calculate Label
	float label = calculateLabel(img, labelPolygon, slidingWindow, imageScaleFactor, showImage);

	// do not train with windows which contain only small parts of the object
	// Only train if sliding window is either a strong positive or a strong negative
    if ( label <= 0 || label > overlapThreshold )
	{		
		float svmLabel = (label > overlapThreshold) ? 1.0 : -1.0;

		if(label > overlapThreshold) 
		{
            // jitter positive images
            additionalImages = doJitter(img, slidingWindow);

            for(int i=0; i<additionalImages.size(); i++)
            {
                transpose (additionalImages[i], additionalImages[i]);
                hog.compute(additionalImages[i], vDescriptor);
                cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);
                labels.push_back(cv::Mat1f(1, 1, svmLabel));
                descriptors.push_back(descriptor);
                positiveTrainingWindows++;
                additionalImages[i].release();
            }

            additionalImages.clear();

			cv::Rect r = cv::Rect(slidingWindow.x / imageScaleFactor, 
				slidingWindow.y / imageScaleFactor, 
				slidingWindow.width / imageScaleFactor, 
				slidingWindow.height / imageScaleFactor);
			predictedSlidingWindows.push_back ( r );
			predictedSlidingWindowWeights.push_back ( label * 2.0 - 1.0 );

		}			
		else 
		{
            //if( (1.0 * rand() / RAND_MAX) < 0.2) // is statistically every 5th time true -> reduce negative training samples
            if(((negativeTrainingWindows+discardedTrainingWindows)%1) == 0 )
            {
				labels.push_back(cv::Mat1f(1, 1, svmLabel));
				descriptors.push_back(descriptor);  

				negativeTrainingWindows++;
            }
			else
			{
				discardedTrainingWindows++;
            }
		}			
    }
    
    img2.release();
}

/// Finish the training. This finalizes the model. Do not call train() afterwards anymore.
void Classifier::finishTraining()
{
	//shuffleTrainingData(descriptors, labels);
	svm.train( descriptors, labels, cv::Mat(), cv::Mat(), svmParams );
	cout << "SVM has been trained" << endl;

	detectionLabels.clear();	
	detectionLabels2.clear();	
	detectionLabels3.clear();	
	detectionPredictions.clear();
	detectionPredictions2.clear();
	detectionPredictions3.clear();
}

void Classifier::saveSVM(string path)
{
    cout << "Saving SVM into file." << endl;
    svm.save(path.c_str());
}

void Classifier::loadSVM(string path)
{
    cout << "Loading SVM from file." << endl;
    svm.load(path.c_str());
}

void Classifier::hardNegativeMine(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor)
{	
    //extract slidingWindow and convert to grayscale
    cv::Mat img2 = img(slidingWindow);
    cvtColor(img2,img2,CV_RGB2GRAY);

	//calculate Feature-Descriptor
	vector<float> vDescriptor;
    transpose(img2, img2);
	hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

	//predict Result
	double prediction = -svm.predict(descriptor, true);
	
	if(prediction > predictionThreshold) //classified as positive
	{
		//calculate Label
		float label = calculateLabel(img, labelPolygon, slidingWindow, imageScaleFactor, false);

		if ( label <= 0.0 ) // classified as positive but no overlap with labelPolygon -> false Positive (hard Negative)
		{		
            float svmLabel = (label > overlapThreshold) ? 1.0 : -1.0;
			labels.push_back(cv::Mat1f(1, 1, svmLabel));
			descriptors.push_back(descriptor);
			hardNegativeMinedWindows++;
		}
	}
}

void Classifier::finishHardNegativeMining()
{
	cout << "Descriptor size: " << descriptors.size() << endl;
	cout << "HardNegativeMining finished" << endl;
	cout << "Retraining SVM ..." << endl;
	//shuffleTrainingData(descriptors, labels);
	svm.train( descriptors, labels, cv::Mat(), cv::Mat(), svmParams );
	cout << "SVM has been retrained after HardNegativeMining" << endl;

	detectionLabels.clear();	
	detectionLabels2.clear();	
	detectionLabels3.clear();	
	detectionPredictions.clear();
	detectionPredictions2.clear();
	detectionPredictions3.clear();
}


/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of having a match inside the sliding window.
///
/// @param img: unknown test image
/// @param slidingWindow: the window section of the image that has to be classified
/// @return: probability of having a match for the target object inside the sliding window section
double Classifier::classify(const cv::Mat& img, cv::Rect slidingWindow, float imageScaleFactor)
{
	//extract slidingWindow and convert to grayscale
	cv::Mat img2 = img(slidingWindow);
	cvtColor(img2,img2,CV_RGB2GRAY);

	//calculate Feature-Descriptor
	vector<float> vDescriptor;
    transpose(img2, img2);
    hog.compute(img2, vDescriptor);
	cv::Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

	//predict Result
	double prediction = -svm.predict(descriptor, true);
	//cout << "Prediction Result:  " << prediction << "( ? > " << predictionThreshold << ")" << endl;
	
	if(prediction > predictionThreshold) 
	{
		cv::Rect r = cv::Rect(slidingWindow.x / imageScaleFactor, 
				slidingWindow.y / imageScaleFactor, 
				slidingWindow.width / imageScaleFactor, 
				slidingWindow.height / imageScaleFactor);
		predictedSlidingWindows.push_back ( r );
		predictedSlidingWindowWeights.push_back ( (float) prediction );
	}

    return prediction;
}

void Classifier::evaluate(double prediction, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor)
{	
	//calculate Label
	cv::Mat emptyMat;
	float label = calculateLabel(emptyMat, labelPolygon, slidingWindow, imageScaleFactor, false);
	classificationLabels.push_back ( label );	
	classificationPredictions.push_back ( prediction );	

	if (prediction > predictionThreshold)
	{
		if(label > overlapThreshold ) 
			cnt_TP += 1;
		else	cnt_FP += 1; 
	}
	else
	{
		if(label > overlapThreshold ) 
			cnt_FN += 1;
		else	cnt_TN += 1;
	}
	cnt_Classified += 1;
}


void Classifier::printEvaluation(bool saveResult)
{	
	cout << "Training:" << endl;
	cout << " -> positiveTrainingWindows:  " << positiveTrainingWindows << endl;
	cout << " -> negativeTrainingWindows:  " << negativeTrainingWindows << endl;
	cout << " -> discardedTrainingWindows: " << discardedTrainingWindows << endl;
	cout << " -> hardNegativeMinedWindows: " << hardNegativeMinedWindows << endl;	
	cout << endl;

	cout << "Classification: " << endl;
	cout << " -> True Positives:           " << cnt_TP << endl;
	cout << " -> False Positives:          " << cnt_FP << endl;
	cout << " -> True Negatives:           " << cnt_TN << endl;
	cout << " -> False Negatives:          " << cnt_FN << endl;
	cout << " -> Classified:               " << cnt_Classified << endl;	
	cout << " -> Accuracy (TP + TN) / All: " << (cnt_TP + cnt_TN) / (1.0 * cnt_Classified) << endl;
	cout << " -> Recall    TP / (TP + FN): " << cnt_TP / (1.0 * (cnt_TP + cnt_FN)) << endl;
	cout << " -> Precision TP / (TP + FP): " << cnt_TP / (1.0 * (cnt_TP + cnt_FP)) << endl;

	if(saveResult)
	{	
		string dir;
		dir = "./ClassificationResults/";
		mkdir(dir.c_str(), 0777);
		dir = ("./ClassificationResults/" + startTime.str()).c_str();
		mkdir(dir.c_str(), 0777);

		std::ofstream outClassificationResult( (dir + "/" + "result_classify_" + startTime.str() + ".csv").c_str() );
		std::ofstream outClassificationResult_share( ("/home/kevin/share/ROC/classify/result_classify_" + startTime.str() + ".csv").c_str());
		for (unsigned i = 0; i < classificationLabels.size(); i++)
		{
			outClassificationResult << ((classificationLabels[i] > overlapThreshold) ? "1.0" : "-1.0");
			outClassificationResult << "\t";
			outClassificationResult << classificationPredictions[i] << endl;

			outClassificationResult_share << ((classificationLabels[i] > overlapThreshold) ? "1.0" : "-1.0");
			outClassificationResult_share << "\t";
			outClassificationResult_share << classificationPredictions[i] << endl;
		}
		outClassificationResult.close();
		outClassificationResult_share.close();

		std::ofstream outDetectionResult( (dir + "/" + "result_detector_" + startTime.str() + ".csv").c_str() );
		std::ofstream outDetectionResult_share( ("/home/kevin/share/ROC/detect/result_detector_" + startTime.str() + ".csv").c_str()  );
		for (unsigned i = 0; i < detectionLabels.size(); i++)
		{
			outDetectionResult << ((detectionLabels[i] > detectionOverlapThreshold) ? "1.0" : "-1.0");
			outDetectionResult << "\t";
			outDetectionResult << detectionPredictions[i] << endl;

			outDetectionResult_share << ((detectionLabels[i] > detectionOverlapThreshold) ? "1.0" : "-1.0");
			outDetectionResult_share << "\t";
			outDetectionResult_share << detectionPredictions[i] << endl;
		}
		outDetectionResult.close();
		outDetectionResult_share.close();


		std::ofstream outDetectionResult2( (dir + "/" + "result_detector2_" + startTime.str() + ".csv").c_str() );
		std::ofstream outDetectionResult2_share( ("/home/kevin/share/ROC/detect/result_detector2_" + startTime.str() + ".csv").c_str()  );
		for (unsigned i = 0; i < detectionLabels2.size(); i++)
		{
			outDetectionResult2 << detectionLabels2[i];
			outDetectionResult2 << "\t";
			outDetectionResult2 << detectionPredictions2[i] << endl;

			outDetectionResult2_share << detectionLabels2[i];
			outDetectionResult2_share << "\t";
			outDetectionResult2_share << detectionPredictions2[i] << endl;
		}
		outDetectionResult2.close();
		outDetectionResult2_share.close();


		std::ofstream outDetectionResult3( (dir + "/" + "result_detector3_" + startTime.str() + ".csv").c_str() );
		std::ofstream outDetectionResult3_share( ("/home/kevin/share/ROC/detect/result_detector3_" + startTime.str() + ".csv").c_str()  );
		for (unsigned i = 0; i < detectionLabels3.size(); i++)
		{
			outDetectionResult3 << detectionLabels3[i];
			outDetectionResult3 << "\t";
			outDetectionResult3 << detectionPredictions3[i] << endl;

			outDetectionResult3_share << detectionLabels3[i];
			outDetectionResult3_share << "\t";
			outDetectionResult3_share << detectionPredictions3[i] << endl;
		}
		outDetectionResult3.close();
		outDetectionResult3_share.close();


		std::ofstream out( (dir + "/" + "_evaluation.txt").c_str() );
		out << "Training:" << endl;
		out << " -> positiveTrainingWindows:  " << positiveTrainingWindows << endl;
		out << " -> negativeTrainingWindows:  " << negativeTrainingWindows << endl;
		out << " -> discardedTrainingWindows: " << discardedTrainingWindows << endl;
		out << " -> hardNegativeMinedWindows: " << hardNegativeMinedWindows << endl;
		out << endl;
		out << "Classification: " << endl;
		out << " -> True Positives:           " << cnt_TP << endl;
		out << " -> False Positives:          " << cnt_FP << endl;
		out << " -> True Negatives:           " << cnt_TN << endl;
		out << " -> False Negatives:          " << cnt_FN << endl;
		out << " -> Classified:               " << cnt_Classified << endl;	
		out << " -> Accuracy (TP + TN) / All: " << (cnt_TP + cnt_TN) / (1.0 * cnt_Classified) << endl;
		out << " -> Recall    TP / (TP + FN): " << cnt_TP / (1.0 * (cnt_TP + cnt_FN)) << endl;
		out << " -> Precision TP / (TP + FP): " << cnt_TP / (1.0 * (cnt_TP + cnt_FP)) << endl;
		out << endl << endl << endl;
		out.close();
	}		
}

void Classifier::shuffleTrainingData(cv::Mat1f  predictionsMatrix, cv::Mat1f labelsMatrix)
{
	std::vector <Mat1f> predictions, labels;
	for (int row = 0; row < predictionsMatrix.rows; row++)
	{
		predictions.push_back(predictionsMatrix.row(row));
		labels.push_back(labelsMatrix.row(row));
	}

	unsigned seed = unsigned ( std::time(0) );
	std::srand ( seed );
	//std::random_shuffle ( vector1.begin(), vector1.end() );
	std::random_shuffle ( predictions.begin(), predictions.end() );

	std::srand ( seed );
	//std::random_shuffle ( vector2.begin(), vector2.end() );
	std::random_shuffle ( labels.begin(), labels.end() );

	for (int row = 0; row < predictionsMatrix.rows; row++)
	{
		predictionsMatrix.row(row) = predictions[row];
		labelsMatrix.row(row) = labels[row];
	}
}

void Classifier::showROC(bool saveROC)
{
	cout << endl << endl;
	cout << "ROC:" << endl;

	int rocSize = 500;
	int TP, TN, FP, FN;
	std::ofstream out;

	cv::Mat3b roc(rocSize + 20, rocSize + 20, cv::Vec3b(255,255,255));
	cv::line(roc, cv::Point2f(10, 10), cv::Point2f(10, rocSize + 10), cv::Scalar(0, 0, 0));
	cv::line(roc, cv::Point2f(10, rocSize + 10), cv::Point2f(rocSize + 10, rocSize + 10), cv::Scalar(0, 0, 0));

	if(saveROC)
	{
		string dir;
		dir = "./ClassificationResults/";
		mkdir(dir.c_str(), 0777);
		dir = ("./ClassificationResults/" + startTime.str()).c_str();
		mkdir(dir.c_str(), 0777);
		out.open( (dir + "/" + "_ROC.txt").c_str() );		
	}
	
	for (float rocThreshold = -1.0; rocThreshold <= 1.0; rocThreshold += 0.1)
	{
		// reset counter values
		TP = TN = FP = FN = 0;
		
		for (unsigned i = 0; i < classificationLabels.size(); i++)
		{
			if (classificationPredictions[i] > rocThreshold)
			{
				if(classificationLabels[i] > overlapThreshold ) 
					TP += 1;
				else	FP += 1; 
			}
			else
			{
				if(classificationLabels[i] > overlapThreshold ) 
					FN += 1;
				else	TN += 1;
			}
		}

		float TP_Rate = TP / (1.0 * TP + FN);
		float FP_Rate = FP / (1.0 * FP + TN);

		if(saveROC)
		{
			
			out << "ROC Threshold: " << rocThreshold << endl;
			out << " -> True Positives:           " << TP << endl;
			out << " -> False Positives:          " << FP << endl;
			out << " -> True Negatives:           " << TN << endl;
			out << " -> False Negatives:          " << FN << endl;	
			out << " -> True Positive Rate    TP / (TP + FN): " << TP_Rate << endl;
			out << " -> False Positive Rate   FP / (FP + TN): " << FP_Rate << endl;
			out << endl;
		}
				
		cv::circle(roc, cv::Point2f(rocSize * FP_Rate + 10, rocSize * (1 - TP_Rate) + 10) , 3, cv::Scalar(0, 0, 255), 1.5, CV_AA);
	}

	// show ROC graphics
	cv::imshow("ROC", roc);


	if(saveROC)
	{
		out.close();
		
		string dir;
		dir = ("./ClassificationResults/" + startTime.str()).c_str();
		cv::imwrite( dir + "/_ROC.jpg", roc );
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

	return (float) overlap;
}


float Classifier::calculateLabel(const cv::Mat& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow, float imageScaleFactor, bool showImage)
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

	return overlapPercentage;	
}



// generate Image with markers for ALL Sliding Windows that are labeled/classified  
void Classifier::generateTaggedResultImage(const cv::Mat& img, string imgName, bool showResult, bool saveResult)
{
	//clone image for drawing shapes
	cv::Mat img_show = img.clone();

	cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8U); 
	cv::Mat segmented;

	//draw sliding predicted sliding windows
	for(int i = 0; i < predictedSlidingWindows.size(); i++){
		rectangle( img_show, predictedSlidingWindows[i], cv::Scalar( 0, 255, 0 ), 2, CV_AA, 0 );

		mask(predictedSlidingWindows[i]) += 10;
		threshold(mask, segmented, 30, 255, cv::THRESH_BINARY);		
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
		cv::imwrite( dir + "/mask_" + imgName, mask );
	}

	// reset sliding window array for next image
	predictedSlidingWindows.clear();
}



void Classifier::evaluateMergedSlidingWindows(const cv::Mat& img, ClipperLib::Path labelPolygon, string imgName, bool showResult, bool saveResult)
{
	double heatmap_threshold = 0.325;
	double area_clippedContourPolygon, area_contourPolygon, area_labelPolygon, TP, FP, overlap, heatmap_max;
	bool targetObjectDetected = false;
	bool targetObjectDetected2 = false;
	bool targetObjectDetected3 = false;
	vector< vector<Point> > contours, contours_thresh;
	vector< vector<Point> > singleContour;
	cv::Mat heatmap_blurred, mask, mask_thresh, img_show, rectCountMap;
	Mat1f heatmap;
	cv::Mat singleContourMask, singleContourHeatmap;
	heatmap = cv::Mat::zeros(img.rows, img.cols, CV_8U); 
	rectCountMap = cv::Mat::zeros(img.rows, img.cols, CV_8U);

	//clone image for drawing shapes
	img_show = img.clone();	

	//draw sliding predicted sliding windows and create heatmap
	for(int i = 0; i < predictedSlidingWindows.size(); i++)
	{		
		if((predictedSlidingWindows[i].x + predictedSlidingWindows[i].width >= img_show.cols) ||
			(predictedSlidingWindows[i].y + predictedSlidingWindows[i].height) >= img_show.rows)
		{		
			continue;
		}		
		rectangle( img_show, predictedSlidingWindows[i], cv::Scalar( 0, 255, 0 ), 1, CV_AA, 0 );      // you need to add a parameter here to set it
		heatmap(predictedSlidingWindows[i]) += (double) 1.0 * predictedSlidingWindowWeights[i];

		// count number of windows
		rectCountMap(predictedSlidingWindows[i]) += 1;
	}

	//calculate Mask Contour	
	cv::GaussianBlur(heatmap, heatmap_blurred, cv::Size(171, 171), 0, 0);       // delete this one
	threshold(heatmap_blurred, mask, 0.0, 255, cv::THRESH_BINARY);
	mask.convertTo(mask,CV_8UC1,255,0);
	findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);


	Rect labelBoundRect;
	if(!labelPolygon.empty())
	{
		//convert labelPolygon to Point-Vector
		vector<Point> labelContour;
		for (int i = 0; i < labelPolygon.size(); i++)
		{
			labelContour.push_back(Point(labelPolygon[i].X, labelPolygon[i].Y));
		}
		vector< vector<Point> > labelContours;
		labelContours.push_back(labelContour);

		// draw labelPolygon		
		//drawContours(img_show, labelContours, -1, cv::Scalar( 255, 0, 0 ), 2, CV_AA);                     // you need to add a parameter here to set it

		// draw bounding rect of labelPolygon
		labelBoundRect = boundingRect(Mat(labelContour));
		rectangle(img_show, labelBoundRect.tl(), labelBoundRect.br(), cv::Scalar( 255, 0, 0 ), 2, 8, 0 );
	}

	for (int i = 0; i < contours.size(); i++)
	{
		//Calculate max value of the heatmap crop of the single contour
		singleContour.clear();
		singleContour.push_back(contours[i]);
		singleContourMask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		singleContourHeatmap = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		drawContours(singleContourMask, singleContour, -1, cv::Scalar( 255, 255, 255 ), CV_FILLED);
		heatmap.copyTo(singleContourHeatmap, singleContourMask);

		Point max_loc;
		cv::minMaxLoc(singleContourHeatmap, NULL, &heatmap_max, NULL, &max_loc);

		int rectCount =  (int)rectCountMap.at<uchar>(max_loc);
		cout << "max_loc: " << max_loc << ", rectCount: " << rectCount << endl;

		cv::GaussianBlur(singleContourHeatmap, singleContourHeatmap, cv::Size(171, 171), 0, 0);
		threshold(singleContourHeatmap, mask_thresh, heatmap_threshold * rectCount, 255, cv::THRESH_BINARY);

		mask_thresh.convertTo(mask_thresh,CV_8UC1,255,0);
		findContours(mask_thresh, contours_thresh, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);

		//draw contours on tagged image
		//drawContours(img_show, contours_thresh, -1, cv::Scalar( 0, 0, 255 ), 2, CV_AA);                   // you need to add a parameter here to set it


		if (contours_thresh.size() > 0)
		{
			// draw bounding rect (detected)
			Rect boundRect = boundingRect( Mat(contours_thresh[0]) );
			Point rect_center(boundRect.x + 0.5 * boundRect.width, boundRect.y + 0.5 * boundRect.height);
			if(1.0 * boundRect.width / boundRect.height < 2.0)
			{
				int newWidth = boundRect.height * 2.0;		
				boundRect = Rect(rect_center.x - 0.5 * newWidth, boundRect.y, newWidth, boundRect.height );
			}
			rectangle(img_show, boundRect.tl(), boundRect.br(), cv::Scalar( 0, 0, 255 ), 2, 8, 0 );

			//draw center point of bounding rect (detected)			
			circle( img_show, rect_center, 10, Scalar( 0, 0, 255 ), -1, 8);  

			//calulate and draw the average center point of the weighted sliding windows (detected)
			Point averageCenterPoint(0, 0);
			double weightSum = 0.0;
			for(int k = 0; k < predictedSlidingWindows.size(); k++)
			{		
				if((predictedSlidingWindows[k].x + predictedSlidingWindows[k].width >= img_show.cols) ||
					(predictedSlidingWindows[k].y + predictedSlidingWindows[k].height) >= img_show.rows)
				{		
					continue;
				}	
				Point centerPoint(predictedSlidingWindows[k].x + 0.5 * predictedSlidingWindows[k].width, predictedSlidingWindows[k].y + 0.5 * predictedSlidingWindows[k].height);
				averageCenterPoint += 1.0 * predictedSlidingWindowWeights[k] * centerPoint;
				weightSum += predictedSlidingWindowWeights[k];
			}
			averageCenterPoint *= (1.0 / weightSum);
 			circle( img_show, averageCenterPoint, 10, Scalar( 0, 255, 0 ), -1, 8); 


			double pred = 1.0 * heatmap_max / rectCount;

			//draw text on image
			ostringstream s;
			s << "Prediction: " << pred << "   (Heatmap-Max: " << heatmap_max << " Rect-Count: " << rectCount << ")";
			putText(img_show, s.str(), cv::Point(10, 40 + i * 80), cv::FONT_HERSHEY_DUPLEX, 1.3, cv::Scalar( 0, 0, 255 ), 2, CV_AA);
			s.str(""); 


			if(!labelPolygon.empty())
			{
				//Calculate Overlap with the contour and the labelPolygon
				ClipperLib::Path contourPolygon;
				for ( int k = 0; k < contours_thresh[0].size(); k++)
				{
					contourPolygon << ClipperLib::IntPoint(contours_thresh[0][k].x, contours_thresh[0][k].y);
				}

				ClipperLib::Paths clippedContourPolygon = clipPolygon(contourPolygon, labelPolygon);
				area_clippedContourPolygon = 0;
				if (clippedContourPolygon.size() > 0)
					area_clippedContourPolygon = abs(Area(clippedContourPolygon[0]));
				area_contourPolygon = abs(Area(contourPolygon));
				area_labelPolygon = abs(Area(labelPolygon));

				TP = area_clippedContourPolygon;
				FP = area_contourPolygon - area_clippedContourPolygon;
				overlap = TP / (area_labelPolygon + FP);

				//new one
				Rect intersection = boundRect & labelBoundRect;
				double areaIntersection = intersection.width * intersection.height;
				double areaLabel = labelBoundRect.width * labelBoundRect.height;
				double areaPrediction = boundRect.width * boundRect.height;
				double areaUnion = areaLabel+areaPrediction-areaIntersection;
				overlap = (double)(areaIntersection/areaUnion);

				//cout << "area_contourPolygon:         " << area_contourPolygon << "   " << abs(area_contourPolygon) << endl;
				//cout << "area_clippedContourPolygon:  " << area_clippedContourPolygon << "   " << abs(area_clippedContourPolygon)  << endl;
				//cout << "area_labelPolygon:           " << area_labelPolygon << "   " << abs(area_labelPolygon)  << endl;
				//cout << "TP:                          " << TP << endl;
				//cout << "FP:                          " << FP << endl;
				cout << "IoU " << (i+1) << "  (TP / (TP + FP + FN)): " << overlap << endl;
				cout << "   -> Heatmap_threshold:          " << heatmap_threshold << endl;
				cout << "   -> Heatmap_max:                " << heatmap_max << endl;
				

				//draw Text
				ostringstream s;				
				s << "IoU: " << overlap << " (-> Label: " << (overlap >= detectionOverlapThreshold) << ")" ;
				putText(img_show, s.str(), cv::Point(10, 80 + i * 80), cv::FONT_HERSHEY_DUPLEX, 1.3, cv::Scalar( 0, 0, 255 ), 2, CV_AA);
				s.str("");


				// detection 1
				if(overlap > detectionOverlapThreshold)
					targetObjectDetected = true;

				detectionLabels.push_back ( overlap );
				detectionPredictions.push_back ( pred );


				// detection 2
				bool detectionInsideLabelBoundRect = rect_center.x >= labelBoundRect.x && rect_center.x <= (labelBoundRect.x + labelBoundRect.width) &&
								rect_center.y >= labelBoundRect.y && rect_center.y <= (labelBoundRect.y + labelBoundRect.height);
				if(detectionInsideLabelBoundRect)     
					targetObjectDetected2 = true;

				detectionLabels2.push_back (detectionInsideLabelBoundRect ? 1.0 : -1.0 );
				detectionPredictions2.push_back ( pred );


				// detection 3
				detectionInsideLabelBoundRect = averageCenterPoint.x >= labelBoundRect.x && averageCenterPoint.x <= (labelBoundRect.x + labelBoundRect.width) &&
								averageCenterPoint.y >= labelBoundRect.y && averageCenterPoint.y <= (labelBoundRect.y + labelBoundRect.height);
				if(detectionInsideLabelBoundRect)     
					targetObjectDetected3 = true;

				detectionLabels3.push_back (detectionInsideLabelBoundRect ? 1.0 : -1.0 );
				detectionPredictions3.push_back ( pred );
			}
		}
	}

	if(!labelPolygon.empty())
	{
		if (!targetObjectDetected)
		{
			detectionLabels.push_back ( 1 );
			detectionPredictions.push_back ( 0 );
		}
		if (!targetObjectDetected2)
		{
			detectionLabels2.push_back ( 1 );
			detectionPredictions2.push_back ( 0 );
		}
		if (!targetObjectDetected3)
		{
			detectionLabels3.push_back ( 1 );
			detectionPredictions3.push_back ( 0 );
		}
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
		//cv::imwrite( dir + "/" + imgName.insert(imgName.length()-5, "_0_heatmap"), heatmap );
		//cv::imwrite( dir + "/" + imgName.insert(imgName.length()-5, "_1_heatmap_blurred"), heatmap_blurred );
		//cv::imwrite( dir + "/" + imgName.insert(imgName.length()-5, "_2_mask.jpg"), mask );
		cv::imwrite( dir + "/" + imgName.insert(imgName.length()-5, "_3_result"), img_show );		
	}

	
	// reset sliding window array for next image
	predictedSlidingWindows.clear();
	predictedSlidingWindowWeights.clear();

	// release images
	heatmap.release();
	heatmap_blurred.release();
	mask.release();
	img_show.release();
	singleContourMask.release();
	singleContourHeatmap.release();
}



// generate Image which shows the clipped polygon and overlap of a single Sliding Window
void Classifier::showTaggedOverlapImage(const cv::Mat& img, ClipperLib::Path labelPolygon, ClipperLib::Path clippedPolygon, cv::Rect slidingWindow, float overlap)
{
	//clone image for drawing shapes
	cv::Mat img_show = img.clone();


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




