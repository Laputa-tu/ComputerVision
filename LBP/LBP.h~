#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class LBP
{
    private:
    template <typename _Tp> void ELBP_(const Mat& src, Mat& dst, int radius, int neighbors);
	void getDescriptor(vector<float>& dst);
        
    public:
        LBP();
        ~LBP();
        void compute(Mat& img, vector<float>& dst);
        
};
