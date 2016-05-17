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

//definitions
#define SLOTH_ZEBRA "zebra.json"
#define SLOTH_SIGN "sign.json"

//functions
void getFilesInDirectory(char *, const char*, int, vector<string> *);
void getFilesInDirectory(char *, const char*, vector<string> *);

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
    getFilesInDirectory(argv[1], SLOTH_ZEBRA, files);

    cout << "Found the following JSON files: " << endl;
    for(vector<string>::iterator it = files->begin(); it != files->end(); ++it)
        cout << "\tFile: " << *it << endl;

    return 0;
}

void getFilesInDirectory(char *path, const char *name, vector<string> *files)
{
    return getFilesInDirectory(path,name,1, files);
}

void getFilesInDirectory(char *path, const char *name, int depth, vector<string> *files)
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
                getFilesInDirectory(newpath, name, depth+1, files);
            else
            {
                cout << "\t";
                for(int i=0; i<depth; i++) cout << " =";
                cout << "> " << path << "/" << dp->d_name;

                if(strcmp(dp->d_name, name) == 0)
                {
                    cout << "\t\t<=== Found JSON File!" << endl;
                    string file_path(newpath);
                    files->push_back(file_path);
                }
                else
                    cout << endl;
            }
        }
    }

    closedir(dirp);
}
