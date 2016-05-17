#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <dirent.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


void searchSlothFilesInDirectory(char *path, char *name, int depth = 1)
{
    DIR *dirp = opendir(path);
    struct dirent *dp;
    struct stat sb;

    if(dirp)
    {
        while ((dp = readdir(dirp)) != NULL)
        {
            // create new path
            char *newpath = (char *) malloc(2 + strlen(path) + strlen(dp->d_name));
            strcpy(newpath, path);
            strcat(newpath, "/");
            strcat(newpath, dp->d_name);

            // skip self and parent
            if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, ".."))
                continue;

            if(stat(newpath, &sb) == 0 && S_ISDIR(sb.st_mode))
                searchSlothFilesInDirectory(newpath, name, depth+1);
            else
            {
                cout << "\t";
                for(int i=0; i<depth; i++) cout << " =";
                cout << "> " << path << "/" << dp->d_name;

                if(strcmp(dp->d_name, name) == 0)
                    cout << "\t\t<=== Found JSON File!" << endl;
                else
                    cout << endl;
            }
        }
    }

    closedir(dirp);
}

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

    // print files
    char sloth[100] = "sloth.json";
    searchSlothFilesInDirectory(argv[1], sloth);

    return 0;
}
