#include "./Main.h"

#include <math.h>

void loadLBPConfiguration()
{
    // training parameters
    originalImageHeight = 1080;
    scale_n_times = 5;
    scaling_factor = 0.8;
    initial_scale = 0.2;
    doHardNegativeMining = false;
    doJitter = false;

    windows_n_rows = 64;
    windows_n_cols = 64;
    step_slide_row = windows_n_rows/4;
    step_slide_col = windows_n_cols/4;

    // classification
    overlapThreshold = 0.5;
    predictionThreshold = 2.0;
    overlapThreshold2 = 0.06;
}


void loadHOGConfiguration()
{
    // training parameters
    originalImageHeight = 1080; 	//1080;
    scale_n_times = 5;              //3;
    scaling_factor = 0.8;           //0.75;
    initial_scale = 0.15;           //0.25;
    doHardNegativeMining = true;
    doJitter = false;

    // sliding window
    windows_n_rows = 64;
    windows_n_cols = 128;
    step_slide_row = windows_n_rows/4;
    step_slide_col = windows_n_cols/4;

    // classification
    overlapThreshold = 0.5;		// label = Percentage of overlap -> 0 to 1.0
    predictionThreshold = 0.5;	// svm prediction: -1 to +1
    overlapThreshold2 = 0.06;	// overlap of the merged-slidingWindow-contour and the labelPolygon
}


int main(int argc, char* argv[])
{
    bool loadSVMFromFile = false;
    //string svm_loadpath = "./SVM_Savings/svm_nice_5_08_015_width128_jitter3_anglestep8.xml"; //_hardnegative
    string svm_loadpath = "./SVM_Savings/svm_2016_7_6__12_11_37.xml"; // lbp
    string svm_savepath = "./SVM_Savings/svm_" + getTimeString() + ".xml";

    char* trainingPath = argv[1];
    char* validationPath = argv[2];
    char* testPath = argv[3];
    vector<JSONImage> trainingSet, validationSet, testSet;
    vector<string> testVideos;
    cnt_TrainingImages = 0;
    cnt_DiscardedTrainingImages = 0;
    imageCounter = 0;

    //load lbp parameters
    loadLBPConfiguration();

    // create model
    Classifier model(overlapThreshold, predictionThreshold, overlapThreshold2, FEATURE_LBPH);

    // check the number of parameters
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <DirTrainingImages zebra.json> [<DirValidationImages zebra.json> <DirTestSet JPG or MP4>]" << endl;
        return WRONG_ARG;
    }

    //get data
    trainingSet = getTrainingSet(trainingPath);
    validationSet = getValidationSet(validationPath);
    testSet = getTestSet(testPath);
    testVideos = getTestVideos(testPath);

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

        if(doHardNegativeMining)
        {
            int res_train_neg = doSlidingOperation(model, trainingSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                                windows_n_cols, step_slide_row, step_slide_col, OPERATE_TRAIN_NEG, originalImageHeight);
            if(res_train_neg != 0)
            {
                cerr << "Error occured during training, errorcode: " << res_train;
                return res_train_neg;
            }

            // finish hard negative mining
            model.finishHardNegativeMining();
        }

        model.saveSVM(svm_savepath);
    }

    // validate
    cout << "Running Validation..." << endl;
    int res_val = doSlidingOperation(model, validationSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_VALIDATE, originalImageHeight);

    if(res_val != 0)
    {
        cerr << "Error occured during validation, errorcode: " << res_val;
        return res_val;
    }

    // classify
    cout << "Running Classification..." << endl;
    int res_test = doSlidingOperation(model, testSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_CLASSIFY, originalImageHeight);
    if(res_test != 0)
    {
        cerr << "Error occured during validation, errorcode: " << res_val;
        return res_test;
    }

    // run classification on videos
    for(int it=0; it<testVideos.size(); it++)
    {
        VideoCapture cap(testVideos.at(it));
        if(!cap.isOpened())
        {
            cout << "Cannot open the video file" << endl;
            return IMG_INVAL;
        }

        cap.set(CV_CAP_PROP_POS_MSEC, 20000); //start the video at 300ms
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
                ClipperLib::Path emptyPolygon;
                int res_test = doSlidingImageOperation(model, frame, emptyPolygon, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                                   windows_n_cols, step_slide_row, step_slide_col, OPERATE_CLASSIFY, originalImageHeight);

                if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
                {
                    cout << "esc key is pressed by user" << endl;
                    break;
                }
            }
        }
    }


    model.printEvaluation(true);
    model.showROC(true);

    waitKey(0);
    return 0;
}

int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight)
{
    Mat image;
    int res;

    // get image from imageSet
    for(int i=0; i<imageSet.size(); i++)
    {
        // read image
        image = imread(imageSet.at(i).getPath(), CV_LOAD_IMAGE_COLOR);
        if(!image.data) // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return IMG_INVAL;
        }

        // do sliding
        res = doSlidingImageOperation(model, image, imageSet.at(i).getLabelPolygon(), scale_n, scale_factor, initial_scale,
                                w_rows, w_cols, step_rows, step_cols, operation, originalImageHeight);

        if(res != 0)
        {
            return res;
        }
    }

    return 0;
}


int doSlidingImageOperation(Classifier &model, Mat frame, ClipperLib::Path labelPolygon, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight)
{
    Mat image, rescaled, rescaled_gray;
    string result_tag;
    float current_scaling;
    bool showTaggedImage = false;
    bool showResult = false;
    bool saveResult = true;

    // check image
    if(frame.rows > 0)
    {
        image = frame;
    }
    else
    {
        return IMG_INVAL;
    }


    if (operation == OPERATE_TRAIN)
    {
        cnt_TrainingImages++;
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
        cvtColor(rescaled, rescaled_gray, CV_RGB2GRAY);

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
                    {
                        model.train(rescaled_gray, rescaled, labelPolygon, windows, current_scaling, doJitter, showTaggedImage);
                        break;
                    }
                    case OPERATE_TRAIN_NEG:
                        model.hardNegativeMine(rescaled_gray, rescaled, labelPolygon, windows, current_scaling);
                        break;
                    case OPERATE_VALIDATE:
                    {
                        double prediction = model.classify(rescaled_gray, rescaled, windows, current_scaling);
                        model.evaluate(prediction, labelPolygon, windows, current_scaling);
                        break;
                    }
                    case OPERATE_CLASSIFY:
                        model.classify(rescaled_gray, rescaled, windows, current_scaling);
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

    ostringstream oss;
    oss << ++imageCounter << ".jpg";

    switch(operation)
    {
        case OPERATE_TRAIN:
            result_tag = "t_";
            //model.evaluateMergedSlidingWindows(image, labelPolygon, result_tag + oss.str(), showResult, saveResult);
            break;
        case OPERATE_CLASSIFY:
            result_tag = "c_";
            model.evaluateMergedSlidingWindows(image, labelPolygon, result_tag + oss.str(), showResult, saveResult);
            break;
        case OPERATE_VALIDATE:
            result_tag = "v_";
            model.evaluateMergedSlidingWindows(image, labelPolygon, result_tag + oss.str(), showResult, saveResult);
            break;
    }

    rescaled.release();
    image.release();

    return 0;
}

vector<JSONImage> getTrainingSet(char *trainingPath)
{
    // get training images
    vector <JSONImage> trainingSet = FileManager::GetJSONImages(trainingPath);
    if(trainingSet.empty())
    {
        cerr << "No training images found." << endl;
    }
    else
    {
        cout << "Found " << trainingSet.size() << " training images." << endl;
    }
    return trainingSet;
}
vector<JSONImage> getValidationSet(char *validationPath)
{
    // get validation images
    vector<JSONImage> validationSet = FileManager::GetJSONImages(validationPath);
    if(validationSet.empty())
    {
        cerr << "No validation images found." << endl;
    }
    else
    {
        cout << "Found " << validationSet.size() << " validation images." << endl;
    }
    return validationSet;
}
vector<JSONImage> getTestSet(char *testPath)
{
    // get test images
    vector<JSONImage> testSet = FileManager::GetImages(testPath);
    if(testSet.empty())
    {
        cerr << "No test images found." << endl;
    }
    else
    {
        cout << "Found " << testSet.size() << " test images." << endl;
    }
    return testSet;
}
vector<string> getTestVideos(char *testPath)
{
    // get test videos from dir
    vector<string> testVideos = FileManager::GetVideosFromDirectory(testPath);
    if(testVideos.empty())
    {
        cerr << "No test videos found." << endl;
    }
    else
    {
        cout << "Found " << testVideos.size() << " test videos." << endl;
    }
    return testVideos;
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

