#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdlib>
#include <pthread.h>

#include <time.h>
#include <stdlib.h>     /* abs */
#include <math.h>       /* pow */

#include <ctime>        // std::time

#include "Helper/FileManager.h"
#include "error.h"

// The VLFeat header files need to be declared external.
extern "C" {
    #include "lib_vlfeat/vl/generic.h"
    #include "lib_vlfeat/vl/lbp.h"
}

#define OPERATE_TRAIN 1
#define OPERATE_CLASSIFY 2
#define OPERATE_VALIDATE 3
#define OPERATE_TRAIN_HARDNEG 4
#define OPERATE_TRAIN_NEG 5

using namespace std;
using namespace cv;

int cnt_TrainingImages, cnt_DiscardedTrainingImages;

vector<JSONImage> getTrainingSet(char *trainingPath);
vector<JSONImage> getValidationSet(char *validationPath);
vector<JSONImage> getTestSet(char *testPath);
vector<string> getTestVideos(char *testPath);
vector<JSONImage> getNegativeSet(char *negativePath);

int doSlidingOperation(Classifier &model, vector<JSONImage> &imageSet, int scale_n, float scale_factor, float initial_scale, int w_rows,
                       int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight, string dir = "./ClassificationResults/");

int doSlidingImageOperation(Classifier &model, Mat frame, ClipperLib::Path labelPolygon, int scale_n, float scale_factor, float initial_scale, int w_rows,
                            int w_cols, int step_rows, int step_cols, const int operation, int originalImageHeight, string dir = "./ClassificationResults/");

int calculateBestSlidingWindow(vector<JSONImage> &imageSet, bool showResult, float initial_scale, int w_rows, int w_cols);

void loadLBPConfiguration();
void loadHOGConfiguration();

static string getTimeString();
static string TimeString;
static int imageCounter;

int train(Classifier &model, bool loadSVMFromFile, string svm_loadpath, string svm_savepath, vector<JSONImage> trainingSet, vector<JSONImage> negativeSet);
int validate(Classifier &model, vector<JSONImage> validationSet);
int classify(Classifier &model, vector<JSONImage> testSet, vector<string> testVideos, string dir = "./ClassificationResults/");
int classify(Classifier &model, vector<JSONImage> testSet, string dir = "./ClassificationResults/");
int classify(Classifier &model, vector<string> testVideos, string dir = "./ClassificationResults/");
void printScaleSteps();
void createVideo(string dir);


// training parameters
int originalImageHeight;
int scale_n_times;
float scaling_factor;
float initial_scale;
bool doHardNegativeMining;
bool doJitter;

// sliding window
int windows_n_rows;
int windows_n_cols;
int step_slide_row;
int step_slide_col;

// classification parameters
float overlapThreshold;
float predictionThreshold;
float overlapThreshold2;

//video output
int ex_video_output;
Size s_video_output;
double fps_video_output;


