#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>
#include <stdlib.h>     /* abs */
#include <math.h>       /* pow */

#include "Helper/FileManager.h"
#include "error.h"

#define OPERATE_TRAIN 1
#define OPERATE_CLASSIFY 2
#define OPERATE_VALIDATE 3


int cnt_TrainingImages, cnt_DiscardedTrainingImages;

int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation);

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    cnt_TrainingImages = 0;
    cnt_DiscardedTrainingImages = 0;
    Mat image, rescaled;
    Classifier model;
    char* trainingPath = argv[1];
    char* validationPath = argv[2];
    char* testPath = argv[3];
    bool saveResult = true;
    vector<JSONImage> trainingSet, validationSet, testSet;

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
    testSet = FileManager::GetImages(testPath);
    if(testSet.empty())
    {
        cerr << "No test images found." << endl;
    }

    // shuffle all images
    //FileManager::ShuffleImages(trainingSet);
    //FileManager::ShuffleImages(validationSet);

    cout << "\nStarting training..." << endl;
    model.startTraining();

    // training parameters
    int scale_n_times = 3;
    float scaling_factor = 0.75;
    float initial_scale = 0.25;
    int originalImageHeight = 1080;
    int windows_n_rows = originalImageHeight * initial_scale * pow(scaling_factor, scale_n_times); //114
    int windows_n_cols = originalImageHeight * initial_scale * pow(scaling_factor, scale_n_times); //114
    windows_n_rows = max(windows_n_rows, 128); // if lower than 128, set to 128
    windows_n_cols = max(windows_n_cols, 128); // if lower than 128, set to 128
    int step_slide_row = windows_n_rows/5;
    int step_slide_col = windows_n_cols/5;

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

    // validate
    cout << "Running Validation..." << endl;
    int res_val = doSlidingOperation(model, trainingSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_VALIDATE);

    if(res_val != 0)
    {
        cerr << "Error occured during validation, errorcode: " << res_val;
        return res_val;
    }

    model.finishHardNegativeMining();

    cout << "Running Classification..." << endl;
    int res_test = doSlidingOperation(model, testSet, scale_n_times, scaling_factor, initial_scale, windows_n_rows,
                                       windows_n_cols, step_slide_row, step_slide_col, OPERATE_CLASSIFY);

    if(res_test != 0)
    {
        cerr << "Error occured during test, errorcode: " << res_test;
        return res_test;
    }

    cout << endl;
    cout << "Used Training Images:      " << cnt_TrainingImages << endl;
    cout << "Discarded Training Images: " << cnt_DiscardedTrainingImages << endl << endl;

    model.printEvaluation(true);
    model.showROC(true);

    waitKey(0);
    return 0;
}


int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation)
{
    Mat image, rescaled, rescaled2;
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
            // check size of LabelPolygon area
            labelPolygonArea = initial_scale * Area(imageSet.at(i).getLabelPolygon());
            if(abs(labelPolygonArea) < 0.5 * slidingWindowArea)
            {
                cnt_DiscardedTrainingImages++;
                cout << "\t--- Discarded image due to small polygon area" << endl;
                //cout << " -> Polygon Area: " << labelPolygonArea << "    Sliding Window Area: " << slidingWindowArea << endl;
                continue; // skip training this image to reduce negative training samples
            }
            else
            {
                cnt_TrainingImages++;
            }
        }

        // read image
        image = imread(imageSet.at(i).getPath(), CV_LOAD_IMAGE_GRAYSCALE);
        if(!image.data) // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return IMG_INVAL;
        }

        //imshow("Image", image);

        rescaled = image;
        resize(rescaled, rescaled, Size(), initial_scale, initial_scale, INTER_CUBIC);
        current_scaling = initial_scale;
        bool reached_row_end = false;
        bool reached_col_end = false;

        for(int j=0; j<scale_n; j++)
        {
            cout << "\tImage: "<<imageSet.at(i).getName() << " (" << rescaled.cols << " x "
                 << rescaled.rows << ", scale " << current_scaling << ")" << endl;

            // build sliding window
            for(int row = 0; row <= rescaled.rows; row += step_rows)
            {
                // check end of rows
                reached_row_end = (rescaled.rows - (row + w_rows) <= 0) ? true : false;
                if(reached_row_end)
                {
                    row = rescaled.rows - w_rows;
                }

                for(int col = 0; col <= rescaled.cols; col += step_cols )
                {
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
                        case OPERATE_VALIDATE:
                            model.hardNegativeMine(rescaled, imageSet.at(i).getLabelPolygon(), windows, current_scaling);
                            break;
                        case OPERATE_CLASSIFY:
                            double pred = model.classify(rescaled, windows, current_scaling);
                            model.evaluate(pred, imageSet.at(i).getLabelPolygon(), windows, current_scaling);
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


            if(j + 1 < scale_n) // only scale if necessary
            {                
                rescaled.release();
                current_scaling = current_scaling*scale_factor;
                resize(image, rescaled, Size(), current_scaling, current_scaling, INTER_CUBIC);
            }


        }

        switch(operation)
        {
            case OPERATE_TRAIN: result_tag = "t_"; break;
            case OPERATE_CLASSIFY: result_tag = "c_"; break;
            case OPERATE_VALIDATE: result_tag = "v_"; break;
        }

        model.generateTaggedResultImage(image, result_tag + imageSet.at(i).getName(), showResult, saveResult);
        rescaled.release();
        image.release();
    }

    return 0;
}
