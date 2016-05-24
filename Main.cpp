#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>
#include <stdlib.h>     /* abs */

#include "Helper/FileManager.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
    // check the number of parameters
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <DirectoyOfTrainingImages>" << endl;
        return 1;
    }

    // print the directory
    cout << "Searching in \"" << argv[1] << "\" ..." << endl;

    // search for directory
    struct stat sb;
    if(stat(argv[1], &sb) != 0 || !S_ISDIR(sb.st_mode))
    {
        cerr << "Error: \"" << argv[1] <<"\" is not a valid directory." << endl;
        return 1;
    }

    // search & get files
    vector<string> files[100];
    FileManager::GetFilesInDirectory(argv[1], SLOTH_ZEBRA, files);

    cout << "Found the following JSON files: " << endl;
    for(vector<string>::iterator it = files->begin(); it != files->end(); ++it)
    {
        cout << "\tFile: " << *it << endl;
    }

    // get images from json
    vector<JSONImage> json_images = FileManager::GetJSONImages(files);

    // start training
    Classifier model;
    model.startTraining();

    cout << "Found " << json_images.size() << " labled images in json files:" << endl;


    Mat image, hsv, rescaled; 

    for(int i=0; i<json_images.size(); i++)
    {
        cout << "\tImage: "<<json_images.at(i).getName() << endl;
	
        // read image        
        image = imread(json_images.at(i).getPath(), CV_LOAD_IMAGE_COLOR);

        if(! image.data ) // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        rescaled = image;
        int scale_n_times = 1;
        float current_scaling = 1;
        float scaling_factor = 1.0;

        for(int j=0; j<scale_n_times; j++)
        {
            current_scaling = current_scaling * scaling_factor;
            resize(rescaled, rescaled, Size(), scaling_factor, scaling_factor, INTER_CUBIC);
            cout << "Width: " << rescaled.cols << endl;
            cout << "Height: " << rescaled.rows << endl;

            // build sliding window
            int windows_n_rows = 128;
            int windows_n_cols = 64;
            int step_slide_row = windows_n_rows/3;
            int step_slide_col = windows_n_cols/3;

            for(int row = 0; row <= rescaled.rows - windows_n_rows; row += step_slide_row)
            {
                for(int col = 0; col <= rescaled.cols - windows_n_cols; col += step_slide_col )
                {
                    Rect windows(col, row, windows_n_cols, windows_n_rows);
                    //Mat Roi = rescaled(windows);

                    //cout << "Pol " << json_images.at(i).getLabelPolygon() << endl;
                    //train
		    bool showTaggedImage = false;
                    model.train(rescaled, json_images.at(i).getLabelPolygon(), windows, current_scaling, showTaggedImage);
                }
            }
        }	
	bool showResult = false;
	bool saveResult = true;
	model.generateTaggedResultImage(image, "t_" + json_images.at(i).getName(), showResult, saveResult);
    }

    cout << "Finishing Training ..." << endl;
    model.finishTraining();

    cout << "Found " << json_images.size() << " labled images in json files:" << endl;
    for(int i=0; i<json_images.size(); i++)
    {
        cout << "\tImage: "<<json_images.at(i).getName() << endl;

        // read image        
        image = imread(json_images.at(i).getPath(), CV_LOAD_IMAGE_COLOR);
	cout << json_images.at(i).getPath() << endl;
        if(! image.data ) // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        rescaled = image;
        int scale_n_times = 1;
        float current_scaling = 1;
        float scaling_factor = 1.0;

        for(int j=0; j<scale_n_times; j++)
        {
            current_scaling = current_scaling * scaling_factor;
            resize(rescaled, rescaled, Size(), scaling_factor, scaling_factor, INTER_CUBIC);
            cout << "Width: " << rescaled.cols << endl;
            cout << "Height: " << rescaled.rows << endl;

            // build sliding window
            int windows_n_rows = 128;
            int windows_n_cols = 64;
            int step_slide_row = windows_n_rows/3;
            int step_slide_col = windows_n_cols/3;

            for(int row = 0; row <= rescaled.rows - windows_n_rows; row += step_slide_row)
            {
                for(int col = 0; col <= rescaled.cols - windows_n_cols; col += step_slide_col )
                {
                    Rect windows(col, row, windows_n_cols, windows_n_rows);
                    //Mat Roi = rescaled(windows);

                    //cout << json_images.at(i).getLabelPolygon() << endl;
                    //train
                    double prediction = model.classify(rescaled, windows, current_scaling);
                    model.evaluate(prediction, json_images.at(i).getLabelPolygon(), windows, current_scaling);

                    //cout << "prediction: " << prediction << endl;
                }
            }
        }
	bool showResult = false;
	bool saveResult = true;
	model.generateTaggedResultImage(image, "c_" + json_images.at(i).getName(), showResult, saveResult);
    }

    bool saveResult = true;
    model.printEvaluation(saveResult);

    waitKey(0);
    return 0;
}
