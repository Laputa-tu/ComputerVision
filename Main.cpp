#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>
#include <stdlib.h>     /* abs */

#include "Helper/FileManager.h"
#include "error.h"

#define OPERATE_TRAIN 1
#define OPERATE_CLASSIFY 2

int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation);

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    Mat image, rescaled;
    Classifier model;
    char* trainingPath = argv[1];
    char* testPath = argv[2];
    bool saveResult = true;
    vector<JSONImage> trainingSet, validationSet, testSet;

    // check the number of parameters
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <DirTrainingImages> <DirTestImages>" << endl;
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
    testSet = FileManager::GetJSONImages(testPath);
    if(testSet.empty())
    {
        cerr << "No test images found." << endl;
        return DAT_INVAL;
    }

    cout << "\nStarting training..." << endl;
    model.startTraining();

    // training parameters
    int windows_n_rows = 1080*0.125; //128
    int windows_n_cols = 1000*0.125;
    int step_slide_row = windows_n_rows/3; //3
    int step_slide_col = windows_n_cols/3; //3
    int scale_n_times = 3;
    float scaling_factor = 0.75;
    float initial_scale = 0.25;

    //train
    int res_train = doSlidingOperation(model, trainingSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                        windows_n_cols, step_slide_row, step_slide_col, OPERATE_TRAIN);
    if(res_train != 0)
    {
        cerr << "Error occured during training, errorcode: " << res_train;
        return res_train;
    }

    cout << "\nFinishing Training ..." << endl;
    model.finishTraining();


    cout << "Classifying..." << endl;
    int res_class = doSlidingOperation(model, testSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_CLASSIFY);

    if(res_class != 0)
    {
        cerr << "Error occured during classification, errorcode: " << res_class;
        return res_class;
    }

    model.printEvaluation(true);

    waitKey(0);
    return 0;
}


int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation)
{
    Mat image, rescaled;
    string result_tag;
    float current_scaling;
    bool showTaggedImage = false;
    bool showResult = false;
    bool saveResult = true;

    for(int i=0; i<imageSet.size(); i++)
    {
        // read image
        image = imread(imageSet.at(i).getPath(), CV_LOAD_IMAGE_GRAYSCALE);
        if(!image.data) // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return IMG_INVAL;
        }

        rescaled = image;
        resize(rescaled, rescaled, Size(), initial_scale, initial_scale, INTER_CUBIC);
        current_scaling = initial_scale;
        for(int j=0; j<scale_n; j++)
        {
            cout << "\tImage: "<<imageSet.at(i).getName() << " (" << rescaled.cols << " x "
                 << rescaled.rows << ", scale " << current_scaling << ")" << endl;

            // build sliding window
            for(int row = 0; row <= rescaled.rows - w_rows; row += step_rows)
            {
                for(int col = 0; col <= rescaled.cols - w_cols; col += step_cols )
                {
                    Rect windows(col, row, w_cols, w_rows);

                    switch (operation)
                    {
                        case OPERATE_TRAIN:
                            model.train(rescaled, imageSet.at(i).getLabelPolygon(), windows, current_scaling, showTaggedImage);
                            break;
                        case OPERATE_CLASSIFY:
                            double prediction = model.classify(rescaled, windows, current_scaling);
                            model.evaluate(prediction, imageSet.at(i).getLabelPolygon(), windows, current_scaling);
                            break;
                    }

                }
            }

            resize(rescaled, rescaled, Size(), scale_factor, scale_factor, INTER_CUBIC);
            current_scaling = current_scaling*scale_factor;
        }

        result_tag = (operation == OPERATE_TRAIN) ? "t_" : "c_";
        model.generateTaggedResultImage(image, result_tag + imageSet.at(i).getName(), showResult, saveResult);
    }

    return 0;
}
