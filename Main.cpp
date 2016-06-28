#include "./Main.h"


int main(int argc, char* argv[])
{
    bool loadSVMFromFile = true;
    string svm_loadpath = "./SVM_Savings/svm_nice_5_08_015_width128_jitter3_anglestep8.xml"; //_hardnegative
    string svm_savepath = "./SVM_Savings/svm_" + getTimeString() + ".xml";

    char* trainingPath = argv[1];
    char* validationPath = argv[2];
    char* testPath = argv[3];
    vector<JSONImage> trainingSet, validationSet, testSet;
    cnt_TrainingImages = 0;
    cnt_DiscardedTrainingImages = 0;

    // training parameters          3 75 15 nächster versuch - initial scale kleiner!
    int originalImageHeight = 1080; 	//1080;
    int scale_n_times = 5; 		//3;
    float scaling_factor = 0.8;	//0.75;
    float initial_scale = 0.15;		//0.25;

    // classification parameters
    overlapThreshold = 0.5;		// label = Percentage of overlap -> 0 to 1.0
    float predictionThreshold = 0.6;	// svm prediction: -1 to +1
    float overlapThreshold2 = 0.06;	// overlap of the merged-slidingWindow-contour and the labelPolygon

    Classifier model(overlapThreshold, predictionThreshold, overlapThreshold2);

    // sliding window
    int windows_n_rows = originalImageHeight * initial_scale * pow(scaling_factor, scale_n_times);
    int windows_n_cols = originalImageHeight * initial_scale * pow(scaling_factor, scale_n_times);
    windows_n_rows = max(windows_n_rows, 128); // if lower than 128, set to 128
    windows_n_cols = max(windows_n_cols, 128); // if lower than 128, set to 128
    windows_n_rows = 64;
    windows_n_cols = 128;
    int step_slide_row = windows_n_rows/4;
    int step_slide_col = windows_n_cols/4;

    // check the number of parameters
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <DirTrainingImages> [<DirValidationImages> <DirTestImages>]" << endl;
        return WRONG_ARG;
    }

    // get images from json
    trainingSet = FileManager::GetJSONImages(trainingPath);
    if(trainingSet.empty())
    {
        cerr << "No training images found." << endl;
        return DAT_INVAL;
    }

    // get images from json
    validationSet = FileManager::GetJSONImages(validationPath);
    if(validationSet.empty())
    {
        cerr << "No validation images found." << endl;
        return DAT_INVAL;
    }

    // get images from json
    /*testSet = FileManager::GetJSONImages(testPath);
    if(testSet.empty())
    {
        cerr << "No test images found." << endl;
    }*/

    // get videos from dir
    vector<string> files = FileManager::GetVideosFromDirectory(testPath);
    if(files.empty())
    {
        cerr << "No test videos found." << endl;
    }

    // print calculated scale steps
    cout << "\nOriginal Image Height: " << originalImageHeight  << endl;
    cout << "\nScale Steps: " << scale_n_times << endl;    
    for (int i = 0; i <= scale_n_times; i++)
    {
        cout << "\tScale Step " << i << " -> Image-Height: " << (originalImageHeight * initial_scale * pow(scaling_factor, i)) << ((i==0) ? " (Initial Scale)" : "") << endl;
    }
    cout << "\nSliding Window Size: " << windows_n_cols << " x " << windows_n_rows << endl;

    // Calculate average bounding box of label-polygons to get the best sliding window size
    cout << "\nCalculating best sliding window size..." << endl;
    calculateBestSlidingWindow(trainingSet, false, initial_scale, windows_n_rows, windows_n_cols);


    cout << "\nStarting training..." << endl;
    model.startTraining(TimeString);

    if(loadSVMFromFile)
    {
        model.loadSVM(svm_loadpath);
    }
    else
    {
        // train
        int res_train = doSlidingOperation(model, trainingSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                            windows_n_cols, step_slide_row, step_slide_col, OPERATE_TRAIN, originalImageHeight);
        if(res_train != 0)
        {
            cerr << "Error occured during training, errorcode: " << res_train;
            return res_train;
        }
        cout << "\nFinishing Training ..." << endl;
        model.finishTraining();

        // do hard negative mining
        int res_train_neg = doSlidingOperation(model, trainingSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                            windows_n_cols, step_slide_row, step_slide_col, OPERATE_TRAIN_NEG, originalImageHeight);
        if(res_train_neg != 0)
        {
            cerr << "Error occured during training, errorcode: " << res_train;
            return res_train_neg;
        }

        // finish hard negative mining
        model.finishHardNegativeMining();
        model.saveSVM(svm_savepath);
    }

    // run classification from videos
    for(int it=0; it<files.size(); it++)
    {
        VideoCapture cap(files.at(it));
        if(!cap.isOpened())
        {
            cout << "Cannot open the video file" << endl;
        }

        cap.set(CV_CAP_PROP_POS_MSEC, 30000); //start the video at 300ms
        double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
        cout << "Frame per seconds : " << fps << endl;
        namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
        int frameCount = 0;

        while(1)
        {
            Mat frame;
            bool bSuccess = cap.read(frame); // read a new frame from video
            if (!bSuccess) //if not success, break loop
            {
               cout << "Cannot read the frame from video file" << endl;
               break;
            }

            if(frameCount++ % 30 != 0)
            {
                //imshow("MyVideo", frame); //show the frame in "MyVideo" window
                int res_test = doSlidingImageOperation(model, frame, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                                   windows_n_cols, step_slide_row, step_slide_col, OPERATE_CLASSIFY, originalImageHeight);

                if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
                {
                    cout << "esc key is pressed by user" << endl;
                    break;
                }
            }
        }
    }

    /*
    cout << "Running Classification..." << endl;
    int res_test = doSlidingOperation(model, testSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_CLASSIFY, originalImageHeight);

    if(res_test != 0)
    {
        cerr << "Error occured during test, errorcode: " << res_test;
        return res_test;
    }

    cout << endl;
    cout << "Used Training Images:      " << cnt_TrainingImages << endl;
    cout << "Discarded Training Images: " << cnt_DiscardedTrainingImages << endl << endl;*/

    // validate
    /*cout << "Running Validation..." << endl;
    int res_val = doSlidingOperation(model, validationSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_VALIDATE, originalImageHeight);

    if(res_val != 0)
    {
        cerr << "Error occured during validation, errorcode: " << res_val;
        return res_val;
    }

    model.printEvaluation(true);
    model.showROC(true);*/

    waitKey(0);
    return 0;
}

int calculateBestSlidingWindow(vector<JSONImage> &imageSet, bool showResult, float initial_scale, int w_rows, int w_cols)
{
	double sum_width = 0;
	double sum_height = 0;
	int count = 0;
    ClipperLib::Path labelPolygon;
	vector<Point> labelPolygonVector;
    float labelPolygonArea;
	Rect boundRect;
    float slidingWindowArea = w_rows * w_cols;

	for(int i = 0; i < imageSet.size(); i++)
    {
        // check size of LabelPolygon area
        labelPolygonArea = initial_scale * Area(imageSet.at(i).getLabelPolygon());
        if(abs(labelPolygonArea) < overlapThreshold * slidingWindowArea)
        {
            // skip training this image to reduce negative training samples
            continue;
        }

        labelPolygon = imageSet.at(i).getLabelPolygon();
        labelPolygonVector.clear();
        for (int k = 0; k < labelPolygon.size(); k++)
        {
            labelPolygonVector.push_back(Point(labelPolygon[k].X, labelPolygon[k].Y));
        }
        boundRect = boundingRect( Mat(labelPolygonVector) );
        sum_width += boundRect.width;
        sum_height += boundRect.height;
        count++;

        if(showResult)
        {
            vector< vector<Point> > contour;
            Mat im = imread(imageSet.at(i).getPath(), CV_LOAD_IMAGE_COLOR);
            contour.clear();
            contour.push_back(labelPolygonVector);
            drawContours(im, contour, -1, cv::Scalar( 255, 0, 0 ), 2, CV_AA);
            rectangle( im, boundRect, Scalar( 0, 255, 255 ), 2, CV_AA );
            imshow(imageSet.at(i).getName(), im);
            waitKey(0);
        }
	} 
	cout << "\tAverage Width:        " << sum_width / count << endl;
	cout << "\tAverage Height:       " << sum_height / count << endl;
	cout << "\tAverage Aspect Ratio: " << sum_width / sum_height << endl;
}


int doSlidingImageOperation(Classifier &model, Mat frame, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight)
{
    Mat image, rescaled;
    string result_tag;
    float current_scaling;
    bool showTaggedImage = false;
    bool showResult = true;
    bool saveResult = false;
    float labelPolygonArea;
    float slidingWindowArea = w_rows * w_cols;
    ClipperLib::Path emptyPath;
    int counter = 0;
    ostringstream counterStr;

    if(frame.rows != 0)
    {
        counter++;

        // read image
        if(!frame.data) // Check for invalid input
        {
            cout <<  "Could not open or find the frame" << std::endl ;
            return IMG_INVAL;
        }

        //imshow("asd", frame);
        //waitKey(27);

        //scale image to defaultHeight
        if(frame.rows != originalImageHeight)
        {
            float defaultScale = 1.0 * originalImageHeight / frame.rows;
            resize(frame, frame, Size(), defaultScale, defaultScale, INTER_CUBIC);
        }
        rescaled = frame;

        resize(rescaled, rescaled, Size(), initial_scale, initial_scale, INTER_CUBIC);
        current_scaling = initial_scale;
        bool reached_row_end = false;
        bool reached_col_end = false;

        for(int j=0; j<=scale_n; j++)
        {
            // build sliding window
            for(int row = 0; row <= rescaled.rows; row += step_rows)
            {
                // check if sliding window is too big for scaled image
                if(w_rows >= rescaled.rows)
                {
                    break;
                }
                // check end of rows
                reached_row_end = (rescaled.rows - (row + w_rows) <= 0) ? true : false;
                if(reached_row_end)
                {
                    row = rescaled.rows - w_rows;
                }

                for(int col = 0; col <= rescaled.cols; col += step_cols )
                {
                    // check if sliding window is too big for scaled image
                    if(w_cols >= rescaled.cols)
                    {
                        break;
                    }
                    // check end of cols
                    reached_col_end = (rescaled.cols - (col + w_cols) <= 0) ? true : false;
                    if(reached_col_end)
                    {
                        col = rescaled.cols - w_cols;
                    }

                    Rect windows(col, row, w_cols, w_rows);

                    switch (operation)
                    {
                        case OPERATE_TRAIN:
                            model.train(rescaled, emptyPath, windows, current_scaling, showTaggedImage);
                            break;
                        case OPERATE_TRAIN_NEG:
                            model.hardNegativeMine(rescaled, emptyPath, windows, current_scaling);
                            break;
                        case OPERATE_VALIDATE:
                        {
                            double prediction = model.classify(rescaled, windows, current_scaling);
                            model.evaluate(prediction, emptyPath, windows, current_scaling);
                            break;
                        }
                        case OPERATE_CLASSIFY:
                            model.classify(rescaled, windows, current_scaling);
                            break;
                    }


                    if(reached_col_end)
                    {
                        //finish
                        break;
                    }
                }

                if(reached_row_end)
                {
                    //finish
                    break;
                }
            }

            // only scale if necessary
            if(j + 1 <= scale_n)
            {
                rescaled.release();
                current_scaling = current_scaling*scale_factor;
                resize(frame, rescaled, Size(), current_scaling, current_scaling, INTER_CUBIC);
            }
        }

        counterStr << counter;


        switch(operation)
        {
            case OPERATE_TRAIN:
                //model.evaluateMergedSlidingWindows(frame, emptyPath, result_tag + counterStr.str(), showResult, saveResult);
                result_tag = "t_";
                break;
            case OPERATE_CLASSIFY:
                result_tag = "c_" + counterStr.str();
                model.evaluateMergedSlidingWindows(frame, emptyPath, result_tag, showResult, saveResult);
                break;
            case OPERATE_VALIDATE:
                model.evaluateMergedSlidingWindows(frame, emptyPath, result_tag, showResult, saveResult);
                result_tag = "v_";
                break;
        }

        rescaled.release();
        //frame.release();
    }

    return 0;
}

int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight)
{
    Mat image, rescaled;
    string result_tag;
    float current_scaling;
    bool showTaggedImage = false;
    bool showResult = false;
    bool saveResult = true;
    float labelPolygonArea;
    float slidingWindowArea = w_rows * w_cols;    

    for(int i=0; i<imageSet.size(); i++)
    {	

        if (operation == OPERATE_TRAIN)
        {
            cnt_TrainingImages++;
            /*
            // check size of LabelPolygon area
            labelPolygonArea = initial_scale * Area(imageSet.at(i).getLabelPolygon());
            if(abs(labelPolygonArea) < overlapThreshold * slidingWindowArea)
            {
                // skip training this image to reduce negative training samples
                cnt_DiscardedTrainingImages++;
                continue;
            }
            else
            {
                cnt_TrainingImages++;
            }*/
        }



        // read image
        image = imread(imageSet.at(i).getPath(), CV_LOAD_IMAGE_COLOR);
        if(!image.data) // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return IMG_INVAL;
        }

        //scale image to defaultHeight
        if(image.rows != originalImageHeight)
        {
            float defaultScale = 1.0 * originalImageHeight / image.rows;
            resize(image, image, Size(), defaultScale, defaultScale, INTER_CUBIC);
        }
        rescaled = image;

        resize(rescaled, rescaled, Size(), initial_scale, initial_scale, INTER_CUBIC);
        current_scaling = initial_scale;
        bool reached_row_end = false;
        bool reached_col_end = false;

        for(int j=0; j<=scale_n; j++)
        {
            // build sliding window
            for(int row = 0; row <= rescaled.rows; row += step_rows)
            {
                // check if sliding window is too big for scaled image
                if(w_rows >= rescaled.rows)
                {
                    break;			
                }
                // check end of rows
                reached_row_end = (rescaled.rows - (row + w_rows) <= 0) ? true : false;
                if(reached_row_end)
                {
                    row = rescaled.rows - w_rows;
                }

                for(int col = 0; col <= rescaled.cols; col += step_cols )
                {
                    // check if sliding window is too big for scaled image
                    if(w_cols >= rescaled.cols)
                    {
                        break;			
                    }
                    // check end of cols
                    reached_col_end = (rescaled.cols - (col + w_cols) <= 0) ? true : false;
                    if(reached_col_end)
                    {
                        col = rescaled.cols - w_cols;
                    }

                    Rect windows(col, row, w_cols, w_rows);
                    
                    switch (operation)
                    {
                        case OPERATE_TRAIN:
                            model.train(rescaled, imageSet.at(i).getLabelPolygon(), windows, current_scaling, showTaggedImage);
                            break;
                        case OPERATE_TRAIN_NEG:
                            model.hardNegativeMine(rescaled, imageSet.at(i).getLabelPolygon(), windows, current_scaling);
                            break;
                        case OPERATE_VALIDATE:
                        {
                            double prediction = model.classify(rescaled, windows, current_scaling);
                            model.evaluate(prediction, imageSet.at(i).getLabelPolygon(), windows, current_scaling);
                            break;
                        }
                        case OPERATE_CLASSIFY:
                            model.classify(rescaled, windows, current_scaling);
                            break;
                    }
                   

                    if(reached_col_end)
                    {
                        //finish
                        break;
                    }
                }

                if(reached_row_end)
                {
                    //finish
                    break;
                }
            }

            // only scale if necessary
            if(j + 1 <= scale_n)
            {                
                rescaled.release();
                current_scaling = current_scaling*scale_factor;
                resize(image, rescaled, Size(), current_scaling, current_scaling, INTER_CUBIC);
            }
        }

        switch(operation)
        {
            case OPERATE_TRAIN:
                //model.evaluateMergedSlidingWindows(image, imageSet.at(i).getLabelPolygon(), result_tag + imageSet.at(i).getName(), showResult, saveResult);
                result_tag = "t_";
                break;
            case OPERATE_CLASSIFY:
                model.evaluateMergedSlidingWindows(image, imageSet.at(i).getLabelPolygon(), result_tag + imageSet.at(i).getName(), showResult, saveResult);
                result_tag = "c_"; break;
            case OPERATE_VALIDATE:
                model.evaluateMergedSlidingWindows(image, imageSet.at(i).getLabelPolygon(), result_tag + imageSet.at(i).getName(), showResult, saveResult);
                result_tag = "v_";
                break;
        }

        rescaled.release();
        image.release();
    }

    return 0;
}

string getTimeString()
{
    if(TimeString.empty())
    {
        ostringstream startTime;
        time_t sTime = time(NULL);
        struct tm *sTimePtr = localtime(&sTime);
        startTime << sTimePtr->tm_year + 1900 << "_"
                  << sTimePtr->tm_mon + 1 << "_"
                  << sTimePtr->tm_mday << "__"
                  << sTimePtr->tm_hour << "_"
                  << sTimePtr->tm_min << "_"
                  << sTimePtr->tm_sec;

        TimeString = startTime.str();
    }

    return TimeString;
}

