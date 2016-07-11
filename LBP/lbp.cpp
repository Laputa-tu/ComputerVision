#include "lbp.hpp"

using namespace cv;


int* uniformLookupTable;

unsigned int rotLeft(unsigned int value, int shift, unsigned int bitCount)
{
    unsigned int bitCountMask = (UINT_MAX >> (sizeof(int) * CHAR_BIT - bitCount));

    return ((value << shift) & bitCountMask) | (value >> (bitCount - shift));
}
unsigned int rotRight(unsigned int value, int shift, unsigned int bitCount)
{
    unsigned int bitCountMask = (UINT_MAX >> (sizeof(int) * CHAR_BIT - bitCount));

    return ((value >> shift)) | ((value << (bitCount - shift)) & bitCountMask);
}


template <typename _Tp>
void lbp::OLBP_(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) > center) << 7;
            code |= (src.at<_Tp>(i-1,j) > center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) > center) << 5;
            code |= (src.at<_Tp>(i,j+1) > center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) > center) << 3;
            code |= (src.at<_Tp>(i+1,j) > center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) > center) << 1;
            code |= (src.at<_Tp>(i,j-1) > center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

template <typename _Tp>
void lbp::ELBP_(const Mat& src, Mat& dst, int radius, int neighbors) {
    neighbors = max(min(neighbors,31),1); // set bounds...
    // Note: alternatively you can switch to the new OpenCV Mat_
    // type system to define an unsigned int matrix... I am probably
    // mistaken here, but I didn't see an unsigned int representation
    // in OpenCV's classic typesystem...
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                dst.at<unsigned int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) && (abs(t-src.at<_Tp>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

//uniform Rotation-Invariant LBP
template <typename _Tp>
void lbp::uRLBP_(const Mat& src, Mat& dst, int& valueRange, int radius, int neighbors, bool useUniformEncoding, bool useRotationInvariance)
{
    neighbors = max(min(neighbors,31),1); // set bounds...
    // Note: alternatively you can switch to the new OpenCV Mat_
    // type system to define an unsigned int matrix... I am probably
    // mistaken here, but I didn't see an unsigned int representation
    // in OpenCV's classic typesystem...
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    Mat maxNeighborValues = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    Mat maxNeighborIndices = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);

    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float neighborValue = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);

                if(maxNeighborValues.at<unsigned int>(i-radius,j-radius) < neighborValue)
                {
                    maxNeighborValues.at<unsigned int>(i-radius,j-radius) = neighborValue;
                    maxNeighborIndices.at<unsigned int>(i-radius,j-radius) = n;
                }

                // we are dealing with floating point precision, so add some little tolerance
                dst.at<unsigned int>(i-radius,j-radius) += ((neighborValue > src.at<_Tp>(i,j)) && (abs(neighborValue-src.at<_Tp>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }

    if(useRotationInvariance)
    {
        // rotation invariance
        int value, shift;
        for(int i = 0; i < dst.rows;i++)
        {
            for(int j = 0; j < dst.cols;j++)
            {
                value = (int) dst.at<unsigned int>(i,j);
                shift = maxNeighborIndices.at<unsigned int>(i,j);
                value = rotRight(value, shift, neighbors);
                dst.at<unsigned int>(i,j) = (unsigned int) value;
            }
        }
    }

    if(useUniformEncoding)
    {
        // convert lbp to uniform lbp
        int lbp_value, uniformValue;
        if(uniformLookupTable == 0)
        {
            createUniformLookupTable(neighbors);
        }
        for(int i = 0; i < dst.rows;i++)
        {
            for(int j = 0; j < dst.cols;j++)
            {
                lbp_value = (int) dst.at<unsigned int>(i,j);
                uniformValue = uniformLookupTable[lbp_value];

                dst.at<unsigned int>(i,j) = (unsigned int) uniformValue;
                //cout << lbp_value << " -> " << dst.at<unsigned int>(i,j) << ",   ";
            }
        }
        valueRange = neighbors * (neighbors - 1) + 2; // only for uniform LBP
    }
    else
    {
        valueRange = pow(2, neighbors);
    }
}


template <typename _Tp>
void lbp::VARLBP_(const Mat& src, Mat& dst, int radius, int neighbors) {
    max(min(neighbors,31),1); // set bounds
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32FC1); //! result
    // allocate some memory for temporary on-line variance calculations
    Mat _mean = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat _delta = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat _m2 = Mat::zeros(src.rows, src.cols, CV_32FC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                _delta.at<float>(i,j) = t - _mean.at<float>(i,j);
                _mean.at<float>(i,j) = (_mean.at<float>(i,j) + (_delta.at<float>(i,j) / (1.0*(n+1)))); // i am a bit paranoid
                _m2.at<float>(i,j) = _m2.at<float>(i,j) + _delta.at<float>(i,j) * (t - _mean.at<float>(i,j));
            }
        }
    }
    // calculate result
    for(int i = radius; i < src.rows-radius; i++) {
        for(int j = radius; j < src.cols-radius; j++) {
            dst.at<float>(i-radius, j-radius) = _m2.at<float>(i,j) / (1.0*(neighbors-1));
        }
    }
}

// now the wrapper functions
void lbp::OLBP(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OLBP_<char>(src, dst); break;
        case CV_8UC1: OLBP_<unsigned char>(src, dst); break;
        case CV_16SC1: OLBP_<short>(src, dst); break;
        case CV_16UC1: OLBP_<unsigned short>(src, dst); break;
        case CV_32SC1: OLBP_<int>(src, dst); break;
        case CV_32FC1: OLBP_<float>(src, dst); break;
        case CV_64FC1: OLBP_<double>(src, dst); break;
    }
}

void lbp::ELBP(const Mat& src, Mat& dst, int radius, int neighbors) {
    switch(src.type()) {
        case CV_8SC1: ELBP_<char>(src, dst, radius, neighbors); break;
        case CV_8UC1: ELBP_<unsigned char>(src, dst, radius, neighbors); break;
        case CV_16SC1: ELBP_<short>(src, dst, radius, neighbors); break;
        case CV_16UC1: ELBP_<unsigned short>(src, dst, radius, neighbors); break;
        case CV_32SC1: ELBP_<int>(src, dst, radius, neighbors); break;
        case CV_32FC1: ELBP_<float>(src, dst, radius, neighbors); break;
        case CV_64FC1: ELBP_<double>(src, dst, radius, neighbors); break;
    }
}
void lbp::uRLBP(const Mat& src, Mat& dst, int& valueRange, int radius, int neighbors, bool useUniformEncoding, bool useRotationInvariance) {
    switch(src.type()) {
        case CV_8SC1: uRLBP_<char>(src, dst, valueRange, radius, neighbors, useUniformEncoding, useRotationInvariance); break;
        case CV_8UC1: uRLBP_<unsigned char>(src, dst, valueRange, radius, neighbors, useUniformEncoding, useRotationInvariance); break;
        case CV_16SC1: uRLBP_<short>(src, dst, valueRange, radius, neighbors, useUniformEncoding, useRotationInvariance); break;
        case CV_16UC1: uRLBP_<unsigned short>(src, dst, valueRange, radius, neighbors, useUniformEncoding, useRotationInvariance); break;
        case CV_32SC1: uRLBP_<int>(src, dst, valueRange, radius, neighbors, useUniformEncoding, useRotationInvariance); break;
        case CV_32FC1: uRLBP_<float>(src, dst, valueRange, radius, neighbors, useUniformEncoding, useRotationInvariance); break;
        case CV_64FC1: uRLBP_<double>(src, dst, valueRange, radius, neighbors, useUniformEncoding, useRotationInvariance); break;
    }
}

void lbp::VARLBP(const Mat& src, Mat& dst, int radius, int neighbors) {
    switch(src.type()) {
        case CV_8SC1: VARLBP_<char>(src, dst, radius, neighbors); break;
        case CV_8UC1: VARLBP_<unsigned char>(src, dst, radius, neighbors); break;
        case CV_16SC1: VARLBP_<short>(src, dst, radius, neighbors); break;
        case CV_16UC1: VARLBP_<unsigned short>(src, dst, radius, neighbors); break;
        case CV_32SC1: VARLBP_<int>(src, dst, radius, neighbors); break;
        case CV_32FC1: VARLBP_<float>(src, dst, radius, neighbors); break;
        case CV_64FC1: VARLBP_<double>(src, dst, radius, neighbors); break;
    }
}

// now the Mat return functions
Mat lbp::OLBP(const Mat& src) { Mat dst; OLBP(src, dst); return dst; }
Mat lbp::ELBP(const Mat& src, int radius, int neighbors) { Mat dst; ELBP(src, dst, radius, neighbors); return dst; }
Mat lbp::uRLBP(const Mat& src, int& valueRange, int radius, int neighbors, bool useUniformEncoding, bool useRotationInvariance) { Mat dst; uRLBP(src, dst, radius, neighbors, useUniformEncoding, useRotationInvariance, valueRange); return dst; }
Mat lbp::VARLBP(const Mat& src, int radius, int neighbors) { Mat dst; VARLBP(src, dst, radius, neighbors); return dst; }






bool lbp::getBit(unsigned b, unsigned index)
{
    return ((b & (1 << index)) != 0);
}

void lbp::createUniformLookupTable(int neighbors)
{
    int tableLength = pow(2, neighbors);
    uniformLookupTable = new int[tableLength];

    int uniform_maxValue = neighbors * (neighbors - 1) + 2;
    int uniformValue = 0;
    for (int tableIndex = 0; tableIndex < tableLength; tableIndex++)
    {
        int oldValue = tableIndex;
        int bitChanges = 0;
        for (int j = 0; j < neighbors - 1; j++)
        {
            bitChanges += ( getBit(oldValue, j) != getBit(oldValue, j + 1) );
        }
        bitChanges += ( getBit(oldValue, neighbors - 1) != getBit(oldValue, 0) );

        if ( bitChanges <= 2 )
        {
            //cout << uniformValue << ", ";
            uniformLookupTable[tableIndex] = uniformValue;
            uniformValue++;
        }
        else
        {
            //cout << uniform_maxValue << ", ";
            uniformLookupTable[tableIndex] = uniform_maxValue;
        }
    }
    //cout << endl << endl;
}
