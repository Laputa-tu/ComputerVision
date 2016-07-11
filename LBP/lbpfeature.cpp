#include <cv.h>
#include <highgui.h>
#include "lbp.hpp"
#include "histogram.hpp"
#include <bitset>

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
    int compute(Mat img_color, double* &descriptor, int radius = 5, int neighbours = 8)
    {
        bool doBlur = false;
        bool doQuantisize = false;
        bool useRotInv = true;
        bool useUniformEncoding = false;
        int cellCount_x = 4;
        int cellCount_y = 2;
        int quantization_factor = 256;
        int cellHistSize;

        Mat img_gray, img_blurred, img_lbp, cell, cellHist, spatialHist;
        vector<Mat> histograms;
        Rect cellRectangle;
        int cellPos_x, cellPos_y, cellWidth, cellHeight;


        cvtColor(img_color, img_gray, CV_BGR2GRAY);

        if(doBlur)
        {
            int sigmaX = 3;
            int sigmaY = 3;
            GaussianBlur(img_gray, img_blurred, Size(0, 0), sigmaX, sigmaY, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
        }
        else
        {
            img_blurred = img_gray;
        }

        lbp::uRLBP(img_blurred, img_lbp, cellHistSize, radius, neighbours, useUniformEncoding, useRotInv);


        //interpretGrayCode(img_lbp);        


        // for each cell, calculate histogram, do rotation invariance, add it to vector
        cellWidth = img_lbp.cols / cellCount_x;
        cellHeight = img_lbp.rows / cellCount_y;
        for (int cellIndex_x = 0; cellIndex_x < cellCount_x; cellIndex_x++)
        {
            for (int cellIndex_y = 0; cellIndex_y < cellCount_y; cellIndex_y++)
            {
                // Setup a rectangle to define your region of interest
                cellPos_x = cellIndex_x * cellWidth;
                cellPos_y = cellIndex_y * cellHeight;
                cellRectangle = Rect(cellPos_x, cellPos_y, cellWidth, cellHeight);

                // Crop the full image to that image contained by the rectangle myROI
                // Note that this doesn't copy the data
                cell = img_lbp(cellRectangle);

                //calculate histogram
                cellHist = lbp::histogram(cell, cellHistSize);

                if(doQuantisize)
                {
                    cellHist = lbp::quantisizeHistogram(cellHist, quantization_factor);
                }

                /*if(doRotInv)
                {
                    cellHist = lbp::createRotationInvariantHistogram(cellHist);
                }*/

                //normalizeHistogram(cellHist, 1000);
                //cout << cellHist;
                histograms.push_back(cellHist);
            }
        }

        spatialHist = lbp::createSpatialHistogram(histograms, cellHistSize);


        //normalize spatial histogram and create the final descriptor
        double maxVal;
        minMaxLoc(spatialHist, NULL, &maxVal, NULL, NULL);
        int desc_size = spatialHist.cols;
        descriptor = new double[desc_size];
        for(int bin = 0; bin < desc_size; bin++)
        {
            // save normalized value (range: 0-1)
            descriptor[bin] = spatialHist.at<int>(0, bin) / maxVal;
        }

        //cout << "Descriptor Size: " << desc_size << endl;
        return desc_size;
    }

    void normalizeHistogram(Mat& hist, int maxValue)
    {
        double oldMaxVal;
        minMaxLoc(hist, NULL, &oldMaxVal, NULL, NULL);

        for(int bin = 0; bin < hist.cols; bin++)
        {
            hist.at<int>(0, bin) = maxValue * hist.at<int>(0, bin) / oldMaxVal;
        }
    }

    void interpretGrayCode(Mat img_lbp)
    {
        for (int col = 0; col < img_lbp.cols; col++)
        {
            for (int row = 0; row < img_lbp.rows; row++)
            {
                int lbp_bitCode = img_lbp.at<int>(row, col);
                string gray = bitset<64>(lbp_bitCode).to_string(); //to binary
                string binary = graytoBinary(gray);
                img_lbp.at<int>(row, col) = (int) (bitset<64>(binary).to_ulong());
            }
        }
    }

    char xor_c(char a, char b)
    {
        return (a == b)? '0': '1';
    }


    char flip(char c)
    {
        return (c == '0')? '1': '0';
    }

    //  function to convert binary string to gray string
    string binarytoGray(string binary)
    {
        string gray = "";

        //  MSB of gray code is same as binary code
        gray += binary[0];

        // Compute remaining bits, next bit is comuted by
        // doing XOR of previous and current in Binary
        for (int i = 1; i < binary.length(); i++)
        {
            // Concatenate XOR of previous bit with current bit
            gray += xor_c(binary[i - 1], binary[i]);
        }

        return gray;
    }



    //  function to convert gray code string to binary string
    string graytoBinary(string gray)
    {
        string binary  = "";

        //  MSB of binary code is same as gray code
        binary += gray[0];

        // Compute remaining bits
        for (int i = 1; i < gray.length(); i++)
        {
            // If current bit is 0, concatenate previous bit
            if (gray[i] == '0')
                binary += binary[i - 1];

            // Else, concatenate invert of previous bit
            else
                binary += flip(binary[i - 1]);
        }

        return binary;
    }

};
