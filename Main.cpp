#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

using namespace std;
using namespace cv;

struct dirent *readdir(DIR *dirp);
struct stat sb;

int main(int argc, char* argv[])
{
    // check the number of parameters
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <DirectoyOfTrainingImages>" << endl;
        return 1;
    }

    // print the directory
    cout << "Searching for \"" << argv[1] << "\" ..." << endl;

    // search for directory

    if(stat(argv[1], &sb) != 0 || !S_ISDIR(sb.st_mode))
    {
        cerr << "Error: \"" << argv[1] <<"\" is not a valid directory." << endl;
        return 1;
    }

    cout << "Directory " << argv[1] << " was found, " << sb.st_nlink << " links:" << endl;
    return 0;
}
