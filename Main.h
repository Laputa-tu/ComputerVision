#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>
#include <stdlib.h>     /* abs */
#include <math.h>       /* pow */

#include <ctime>        // std::time

#include "Helper/FileManager.h"
#include "error.h"

#define OPERATE_TRAIN 1
#define OPERATE_CLASSIFY 2
#define OPERATE_VALIDATE 3
#define OPERATE_TRAIN_NEG 4

using namespace std;
using namespace cv;

int cnt_TrainingImages, cnt_DiscardedTrainingImages;

int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight);

int doSlidingImageOperation(Classifier &model, Mat frame, int scale_n, float scale_factor,
                       float initial_scale, int w_rows, int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight);

int calculateBestSlidingWindow(vector<JSONImage> &imageSet, bool showResult, float initial_scale, int w_rows, int w_cols);

static string getTimeString();
static string TimeString;

float overlapThreshold;


