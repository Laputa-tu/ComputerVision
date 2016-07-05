#include <cv.h>
#include <highgui.h>
#include "lbp.hpp"
#include "histogram.hpp"

using namespace cv;

class LBPFeature
{
public:
    /// <summary> Computes the feature vector of an ELBP's spatial histogram</summary>
    /// <param name="image"> image on which the elbp is computed </param>
    /// <param name="descriptor"> feature vecture which contains results after computation</param>
    /// <param name="radius"> radius that will be used in the elbp </param>
    /// <param name="neighbours"> neighbours that will be used in the elbp </param>
    /// <param name="cellSize"> the cellsize is used for the partial histograms that are then pushed into the spatial histogram</param>
    /// <param name="cellSize"> overlap of the cells </param>
    /// <returns> Returns the size of the feature vector </returns>
    int compute(Mat image, double* &descriptor, int radius = 1, int neighbours = 8, int cellSize = 30, int overlap = 0)
    {
        Mat dst; //image after preproc.
        Mat lbp; //lbp image
        Mat sp_hist; // for spatial histogram

        cvtColor(image, dst, CV_BGR2GRAY);
        //GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea

        // do lbp
        lbp::ELBP(dst, lbp, radius, neighbours);
        normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);

        //imshow("image", image);
        //imshow("Display window", lbp);
        //waitKey(0);

        Size window(cellSize, cellSize);
        lbp::spatial_histogram(lbp, sp_hist, 255, window, overlap);

        // get max value for normalization
        double maxVal;
        minMaxLoc(sp_hist, NULL, &maxVal, NULL, NULL);

        int desc_size = sp_hist.cols;
        descriptor = new double[desc_size];
        for(int i=0; i<desc_size; i++)
        {
            // save normalized value (range: 0-1)
            descriptor[i] = sp_hist.at<int>(0, i)/maxVal;
        }

        return desc_size;
    }

    int doLBP(int argc, char *argv[])
    {
        int deviceId = 0;
        if(argc > 1)
            deviceId = atoi(argv[1]);

        VideoCapture cap(deviceId);

        if(!cap.isOpened()) {
            cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
            return -1;
        }

        // initial values
        int radius = 1;
        int neighbors = 8;

        // windows
        namedWindow("original",CV_WINDOW_AUTOSIZE);
        namedWindow("lbp",CV_WINDOW_AUTOSIZE);

        // matrices used
        Mat frame; // always references the last frame
        Mat dst; // image after preprocessing
        Mat lbp; // lbp image

        // just to switch between possible lbp operators
        vector<string> lbp_names;
        lbp_names.push_back("Extended LBP"); // 0
        lbp_names.push_back("Fixed Sampling LBP"); // 1
        lbp_names.push_back("Variance-based LBP"); // 2
        int lbp_operator=0;

        bool running=true;
        while(running) {
            cap >> frame;
            cvtColor(frame, dst, CV_BGR2GRAY);
            GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
            // comment the following lines for original size
            resize(frame, frame, Size(), 0.5, 0.5);
            resize(dst,dst,Size(), 0.5, 0.5);
            //
            switch(lbp_operator) {
            case 0:
                lbp::ELBP(dst, lbp, radius, neighbors); // use the extended operator
                break;
            case 1:
                lbp::OLBP(dst, lbp); // use the original operator
                break;
            case 2:
                lbp::VARLBP(dst, lbp, radius, neighbors);
                break;
            }
            // now to show the patterns a normalization is necessary
            // a simple min-max norm will do the job...
            normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);

            imshow("original", frame);
            imshow("lbp", lbp);

            char key = (char) waitKey(20);

            // exit on escape
            if(key == 27)
                running=false;

            // to make it a bit interactive, you can increase and decrease the parameters
            switch(key) {
            case 'q': case 'Q':
                running=false;
                break;
            // lower case r decreases the radius (min 1)
            case 'r':
                radius-=1;
                radius = std::max(radius,1);
                cout << "radius=" << radius << endl;
                break;
            // upper case r increases the radius (there's no real upper bound)
            case 'R':
                radius+=1;
                radius = std::min(radius,32);
                cout << "radius=" << radius << endl;
                break;
            // lower case p decreases the number of sampling points (min 1)
            case 'p':
                neighbors-=1;
                neighbors = std::max(neighbors,1);
                cout << "sampling points=" << neighbors << endl;
                break;
            // upper case p increases the number of sampling points (max 31)
            case 'P':
                neighbors+=1;
                neighbors = std::min(neighbors,31);
                cout << "sampling points=" << neighbors << endl;
                break;
            // switch between operators
            case 'o': case 'O':
                lbp_operator = (lbp_operator + 1) % 3;
                cout << "Switched to operator " << lbp_names[lbp_operator] << endl;
                break;
            case 's': case 'S':
                imwrite("original.jpg", frame);
                imwrite("lbp.jpg", lbp);
                cout << "Screenshot (operator=" << lbp_names[lbp_operator] << ",radius=" << radius <<",points=" << neighbors << ")" << endl;
                break;
            default:
                break;
            }

        }
    return 0; // success
    }

};
