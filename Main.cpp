#include "./Main.h"

void loadLBPConfiguration()
{
    // training parameters
    originalImageHeight = 1080;
    scale_n_times = 3;
    scaling_factor = 0.8;
    initial_scale = 0.225;
    doHardNegativeMining = false;
    doJitter = false;

    windows_n_rows = 96;
    windows_n_cols = 192;
    step_slide_row = windows_n_rows / 4;
    step_slide_col = windows_n_cols / 4;

    // classification
    overlapThreshold = 0.5;
    predictionThreshold = 0.5;
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
    step_slide_row = windows_n_rows / 4;
    step_slide_col = windows_n_cols / 4;

    // classification
    overlapThreshold = 0.5;		// label = Percentage of overlap -> 0 to 1.0
    predictionThreshold = 0.5;	// svm prediction: -1 to +1
    overlapThreshold2 = 0.06;	// overlap of the merged-slidingWindow-contour and the labelPolygon
}

int main(int argc, char* argv[])
{
    bool loadSVMFromFile = true;
    string svm_loadpath = "./SVM_Savings/svm_final.xml"; // lbp
    string svm_savepath = "./SVM_Savings/svm_" + getTimeString() + ".xml";
    string videoPath = "./ClassificationResults/Videos/";

    char* trainingPath = (argc >= 2) ? argv[1] : NULL;
    char* validationPath = (argc >= 3) ? argv[2] : NULL;
    char* testPath = (argc >= 4) ? argv[3] : NULL;
    char* negativePath = (argc >= 5) ? argv[4] : NULL;
    char* outputVideoPath = (argc >= 6) ? argv[5] : NULL;

    vector<JSONImage> trainingSet, validationSet, testSet, negativeSet;
    vector<string> testVideos;
    cnt_TrainingImages = 0;
    cnt_DiscardedTrainingImages = 0;
    imageCounter = 0;

    // check the number of parameters
    if(argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <DirTrainingImages zebra.json> "
             <<"[<DirValidationImages zebra.json> <DirTestSet .JPG or .MP4> "
             <<"<DirNegativeSet .jpg> <OutputVideoToMerge .MP4>]" << endl;
        return WRONG_ARG;
    }

    //load lbp parameters
    loadLBPConfiguration();

    // create model
    Classifier model(overlapThreshold, predictionThreshold, overlapThreshold2, FEATURE_LBPH);

    //get data
    trainingSet = getTrainingSet(trainingPath);
    validationSet = getValidationSet(validationPath);
    testSet = getTestSet(testPath);
    testVideos = getTestVideos(testPath);
    negativeSet = getNegativeSet(negativePath);
    outputVideoToMerge = getTestVideos(outputVideoPath, true);

    printScaleSteps();
    calculateBestSlidingWindow(trainingSet, false, initial_scale, windows_n_rows, windows_n_cols);

    //train
    int trainingResult = train(model, loadSVMFromFile, svm_loadpath, svm_savepath, trainingSet, negativeSet);
    if(trainingResult != 0)
    {
        return trainingResult;
    }

    // validate
    int validationResult = validate(model, validationSet);
    if(validationResult != 0)
    {
        return validationResult;
    }

    //classify
    int classificationResult = classify(model, testSet, testVideos, videoPath);
    if(classificationResult != 0)
    {
        return classificationResult;
    }

    // finish
    if(testVideos.size() > 0)
    {
        string path = videoPath+getTimeString();
        createVideo(path);
        FileManager::RemoveDirectory(path.c_str());
    }

    model.printEvaluation(true);
    model.showROC(true);
    return 0;
}

int train(Classifier &model, bool loadSVMFromFile, string svm_loadpath, string svm_savepath, vector<JSONImage> trainingSet, vector<JSONImage> negativeSet)
{
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

        // train add. negative samples
        int res_train_neg = doSlidingOperation(model, negativeSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                               windows_n_cols, step_slide_row, step_slide_col, OPERATE_TRAIN_NEG, originalImageHeight);
        if(res_train_neg != 0)
        {
            cerr << "Error occured during training, errorcode: " << res_train_neg;
            return res_train_neg;
        }

        cout << "\nFinishing Training ..." << endl;
        model.finishTraining();

        if(doHardNegativeMining)
        {
            int res_train_neg = doSlidingOperation(model, trainingSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                                windows_n_cols, step_slide_row, step_slide_col, OPERATE_TRAIN_HARDNEG, originalImageHeight);
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

    return 0;
}

int validate(Classifier &model, vector<JSONImage> validationSet)
{
    cout << "Running Validation..." << endl;
    int validationResult = doSlidingOperation(model, validationSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_VALIDATE, originalImageHeight);
    if(validationResult != 0)
    {
        cerr << "Error occured during validation, errorcode: " << validationResult;
        return validationResult;
    }
    else
    {
        return 0;
    }
}

int classify(Classifier &model, vector<JSONImage> testSet, string dir)
{
    int res_test = doSlidingOperation(model, testSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_CLASSIFY, originalImageHeight, dir);
    if(res_test != 0)
    {
        cerr << "Error occured during validation, errorcode: " << res_test;
        return res_test;
    }

    return 0;
}

int classify(Classifier &model, vector<string> testVideos, string dir)
{
    bool mergeVideo = (!outputVideoToMerge.empty()) ? true : false;

    // run classification on videos
    for(int it=0; it<testVideos.size(); it++)
    {
        VideoCapture cap(testVideos.at(it));
        if(!cap.isOpened())
        {
            cout << "Cannot open the video file" << endl;
            return IMG_INVAL;
        }

        cap.set(CV_CAP_PROP_POS_MSEC, 0); //start the video at 300ms

        //save video codec, size and fps for output
        ex_video_output = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));        // Get Codec Type- Int form
        fps_video_output = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
        int frameCount = 0;

        VideoCapture capToMerge;
        if(mergeVideo)
        {
            capToMerge = VideoCapture(outputVideoToMerge.at(it));
            if(!capToMerge.isOpened())
            {
                cout << "Cannot open the video file (to merge)." << endl;
                return IMG_INVAL;
            }

            capToMerge.set(CV_CAP_PROP_POS_MSEC, 0); //start the video at 300ms
        }

        while(1)
        {
            Mat frame;
            bool bSuccess = cap.read(frame); // read a new frame from video
            if (!bSuccess) //if not success, break loop
            {
               cout << "Cannot read the frame from video file" << endl;
               break;
            }

            if(mergeVideo)
            {
                bSuccess = capToMerge.read(frameToMerge);
                if (!bSuccess) //if not success, break loop
                {
                   cout << "Cannot read the frame from video file (to merge)." << endl;
                   break;
                }
            }

            ClipperLib::Path emptyPolygon;

            if((frameCount++ % 1) == 0)
            {
                int res_test = doSlidingImageOperation(model, frame, emptyPolygon, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                                   windows_n_cols, step_slide_row, step_slide_col, OPERATE_CLASSIFY, originalImageHeight, dir, mergeVideo);
            }
        }
    }

    return 0;
}

int classify(Classifier &model, vector<JSONImage> testSet, vector<string> testVideos, string dir)
{
    cout << "Running Classification..." << endl;
    int res_pic, res_vid;/*
    res_pic = classify(model, testSet);
    if(res_pic != 0)
    {
        return res_pic;
    }*/

    res_vid = classify(model, testVideos, dir);
    if(res_vid != 0)
    {
        return res_vid;
    }

    return 0;
}

void printScaleSteps()
{
    // print calculated scale steps
    cout << "\nOriginal Image Height: " << originalImageHeight  << endl;
    cout << "\nScale Steps: " << scale_n_times << endl;
    for (int i = 0; i <= scale_n_times; i++)
    {
        cout << "\tScale Step " << i << " -> Image-Height: " << (originalImageHeight * initial_scale * pow(scaling_factor, i)) << ((i==0) ? " (Initial Scale)" : "") << endl;
    }
    cout << "\nSliding Window Size: " << windows_n_cols << " x " << windows_n_rows << endl;
}

int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor, float initial_scale, int w_rows, int w_cols,
                       int step_rows, int step_cols, const int operation, int originalImageHeight, string dir)
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


int doSlidingImageOperation(Classifier &model, Mat frame, ClipperLib::Path labelPolygon, int scale_n, float scale_factor, float initial_scale, int w_rows,
                            int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight, string dir, bool mergeVideo)
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
                    {
                        model.trainNegativeSample(rescaled_gray, rescaled, windows);
                        break;
                    }
                    case OPERATE_TRAIN_HARDNEG:
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

    // add leading zeros for asc. order
    ostringstream oss;
    oss << setw(5) << setfill('0') << ++imageCounter << ".jpg";

    switch(operation)
    {
        case OPERATE_TRAIN:
            result_tag = "t_";
            //model.evaluateMergedSlidingWindows(image, labelPolygon, result_tag + oss.str(), showResult, saveResult);
            break;
        case OPERATE_CLASSIFY:
        {
            result_tag = "c_";
            if(mergeVideo)
            {
                image = frameToMerge;
                //scale image to defaultHeight
                if(image.rows != originalImageHeight)
                {
                    float defaultScale = 1.0 * originalImageHeight / image.rows;
                    resize(image, image, Size(), defaultScale, defaultScale, INTER_CUBIC);
                }
            }
            model.evaluateMergedSlidingWindows(image, labelPolygon, result_tag + oss.str(), showResult, saveResult, dir);
            break;
        }
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
    vector<JSONImage> testSet = FileManager::GetImages(testPath, IMAGE_JPG);
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
vector<JSONImage> getNegativeSet(char *negativePath)
{
    // get test images
    vector<JSONImage> negativeSet = FileManager::GetImages(negativePath, IMAGE_JPG);
    if(negativeSet.empty())
    {
        cerr << "No (additional) negative images found." << endl;
    }
    else
    {
        cout << "Found " << negativeSet.size() << " negative training images." << endl;
    }
    return negativeSet;
}

vector<string> getTestVideos(char *testPath, bool isVideoToMerge)
{
    // get test videos from dir
    vector<string> testVideos = FileManager::GetVideosFromDirectory(testPath);
    if(testVideos.empty())
    {
        if(isVideoToMerge)
        {
             cout << "No video found to merge." << endl;
        }
        else
        {
             cout << "No test videos found." << endl;
        }
    }
    else
    {
        if(isVideoToMerge)
        {
             cout << "Found video to merge." << endl;
        }
        else
        {
             cout << "Found " << testVideos.size() << " test videos." << endl;
        }

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

    cout << "\nCalculating best sliding window size..." << endl;
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

void createVideo(string dir)
{
    cout << "Creating output video ..." << endl;
    Mat image;
    Size s_video;

    char * path = const_cast<char*> ( dir.c_str() );
    vector<string> images = FileManager::GetImageFilesFromDirectory(path);

    string file_path;
    size_t found;
    string file_name;
    for(vector<string>::iterator it = images.begin(); it != images.end(); ++it)
    {
        // get directory
        file_path = *it;
        found = file_path.find_last_of("/\\");
        file_name = file_path.substr(found+1);

        image = imread(file_path, CV_LOAD_IMAGE_COLOR);
        if(!image.empty())
        {
            s_video = Size(image.cols, image.rows);
            break;
        }
    }

    // write video
    VideoWriter outputVideo;
    string name = dir+getTimeString()+".avi";
    outputVideo.open(name.c_str(),CV_FOURCC('M', 'P', '4', '2') , fps_video_output, s_video, true);

    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex_video_output & 0XFF) , (char)((ex_video_output & 0XFF00) >> 8),
                  (char)((ex_video_output & 0XFF0000) >> 16),(char)((ex_video_output & 0XFF000000) >> 24), 0};

    cout << "\tOutput frame resolution: Width=" << s_video.width << "  Height=" << s_video.height << endl;
    cout << "\tOutput codec type: " << EXT << endl;

    if (!outputVideo.isOpened())
    {
        cerr  << "Could not open the output video for write: " << name.c_str() << endl;
        return;
    }

    std::sort(images.begin(), images.end());
    for(vector<string>::iterator it = images.begin(); it != images.end(); ++it)
    {
        // get directory
        file_path = *it;
        found = file_path.find_last_of("/\\");
        file_name = file_path.substr(found+1);

        image = imread(file_path, CV_LOAD_IMAGE_COLOR);
        outputVideo.write(image);
        image.release();
    }

    outputVideo.release();
}

