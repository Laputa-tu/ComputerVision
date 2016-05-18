#ifdef _MSC_VER
#include <boost/config/compiler/visualc.hpp>
#endif
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <cassert>
#include <exception>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <dirent.h>
#include <iostream>
#include <string>

#include "HOG/classifier.h"

using namespace std;
using namespace cv;

//definitions
#define SLOTH_ZEBRA "zebra.json"
#define SLOTH_SIGN "sign.json"

//functions
void getFilesInDirectory(char *, const char*, int, vector<string> *);
void getFilesInDirectory(char *, const char*, vector<string> *);
vector<double> StringToVector(string);

class JSONImage
{
    private:
        string name, path;
        vector<double> xn;
        vector<double> yn;
    public:
        JSONImage(void)
        {

        }

        ~JSONImage(void)
        {

        }

        void setXn(string xn_string)
        {
            this->xn = StringToVector(xn_string);
        }

        void setYn(string yn_string)
        {
            this->yn = StringToVector(yn_string);
        }

        vector<double> getXn()
        {
            return this->xn;
        }

        vector<double> getYn()
        {
            return this->yn;
        }

        void printXn()
        {
            for(int i=0; i<xn.size(); i++)
            {
                cout.precision(17);
                cout << xn.at(i) << "; ";
            }

            cout << endl;
        }

        void printYn()
        {
            for(int i=0; i<yn.size(); i++)
            {
                cout.precision(17);
                cout << yn.at(i) << "; ";
            }

            cout << endl;
        }

        void setName(string name)
        {
            this->name = name;
        }
        string getName()
        {
            return this->name;
        }

        void setPath(string path)
        {
            this->path = path;
        }
        string getPath()
        {
            return this->path;
        }

        bool hasPolygon()
        {
            if(xn.size() > 0 && yn.size() > 0)
                return true;
            else
                return false;
        }

        ClipperLib::Path getLabelPolygon()
        {
            ClipperLib::Path labelPolygon;
            for(int i=0; i<xn.size(); i++)
            {
                labelPolygon << ClipperLib::IntPoint(xn.at(i), yn.at(i));
            }
            return labelPolygon;
        }
};

// more functions
vector<JSONImage> getJSONImages(vector<string> *);



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
    {
        cout << "\tFile: " << *it << endl;
    }

    // get images from json
    vector<JSONImage> json_images = getJSONImages(files);
    cout << "Found " << json_images.size() << " labled images in json files:" << endl;
    for(int i=0; i<json_images.size(); i++)
    {
        cout << "\tImage: "<<json_images.at(i).getName() << endl;

        // read image
        Mat image, rescaled;
        image = imread(json_images.at(i).getPath(), CV_LOAD_IMAGE_COLOR);

        if(! image.data ) // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        // start training
        Classifier model;
        model.startTraining();

        rescaled = image;
        int scale_n_times = 3;
        float current_scaling = 1;
        float scaling_factor = 0.5;

        for(int i=0; i<scale_n_times; i++)
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

                    cout << json_images.at(i).getLabelPolygon() << endl;
                    //train
                    //model.train(rescaled, json_images.at(i).getLabelPolygon(), windows);
                }
            }
        }
    }

    waitKey(0);
    return 0;
}

/*
 *
 * void Classifier::train(const cv::Mat3b& img, ClipperLib::Path labelPolygon, cv::Rect slidingWindow)
{
    /*
    ClipperLib::Path labelPolygon;
    labelPolygon << IntPoint(0, 0) << IntPoint(70, 0) << IntPoint(100, 60) << IntPoint(70, 100) << IntPoint(0, 50);
    //slidingWindow << IntPoint(20, 20) << IntPoint(120, 20) << IntPoint(120, 80) << IntPoint(20, 80);
    cv::Rect slidingWindow = cv::Rect(0, 0, 64, 128);
    */


vector<JSONImage> getJSONImages(vector<string> *files)
{
    vector<JSONImage> imageList;

    // collect images for each json file
    for(vector<string>::iterator it = files->begin(); it != files->end(); ++it)
    {
        // get directory
        string file_path = *it;
        int index = file_path.find(SLOTH_ZEBRA);
        string file_directory = file_path.substr(0, index);

        try
        {
            boost::property_tree::ptree pt;
            boost::property_tree::read_json(*it, pt);

            BOOST_FOREACH(boost::property_tree::ptree::value_type& v, pt)
            {
                JSONImage currentImage;
                BOOST_FOREACH(boost::property_tree::ptree::value_type& i, v.second)
                {
                    // get and print name
                    string name = i.second.get_value("filename");
                    currentImage.setName(name);
                    currentImage.setPath(file_directory + name);

                    // get labeled polygons
                    BOOST_FOREACH(boost::property_tree::ptree::value_type& j, i.second)
                    {
                        BOOST_FOREACH(boost::property_tree::ptree::value_type& k, j.second)
                        {
                            string xn = k.second.get_value("xn");
                            string yn = k.second.get_value("yn");

                            if(!xn.empty() && !yn.empty())
                            {
                                currentImage.setXn(xn);
                                currentImage.setYn(yn);
                            }
                        }
                    }
                }

                if(currentImage.hasPolygon())
                {
                    imageList.push_back(currentImage);
                }
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    return imageList;
}

vector<double> StringToVector(string str)
{
    vector<double> vect;
    stringstream ss(str);
    double i;

    while (ss >> i)
    {
        vect.push_back(i);

        if (ss.peek() == ';')
            ss.ignore();
    }

    return vect;
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
