#include "FileManager.h"

#define MAX_NUMBER_FILES 100000

bool FileManager::IsValidDirectory(char *path)
{
    struct stat sb;
    if(stat(path, &sb) != 0 || !S_ISDIR(sb.st_mode))
    {
        return false;
    }
    else
    {
        return true;
    }
}

void FileManager::GetFilesInDirectory(char *path, const char *name, vector<string> *files)
{
    return GetFilesInDirectory(path,name,1, files);
}

void FileManager::GetFilesInDirectory(char *path, const char *name, int depth, vector<string> *files)
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
                GetFilesInDirectory(newpath, name, depth+1, files);
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

vector<JSONImage> FileManager::GetJSONImages(char* path)
{
    vector<string> files[MAX_NUMBER_FILES];
    vector<JSONImage> imageList;

    // search for directory
    if(!FileManager::IsValidDirectory(path))
    {
        cerr << "Error: \"" << path <<"\" is not a valid directory, error code " << DIR_INVAL << endl;
        return imageList;
    }

    // search & get files
    cout << "Searching in \"" << path << "\":" << endl;
    FileManager::GetFilesInDirectory(path, SLOTH_ZEBRA, files);

    cout << "\nFound the following JSON files:" << endl;
    for(vector<string>::iterator it = files->begin(); it != files->end(); ++it)
    {
        cout << "\tFile: " << *it << endl;
    }

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
                    if(i.first == "filename")
                    {
                        string name = i.second.get_value<string>();
                        currentImage.setName(name);
                        currentImage.setPath(file_directory + name);
                    }

                    if(i.first == "annotations")
                    {
                        BOOST_FOREACH(boost::property_tree::ptree::value_type& j, i.second)
                        {
                            BOOST_FOREACH(boost::property_tree::ptree::value_type& h, j.second)
                            {

                                if(h.first == "xn")
                                {
                                    string xn = h.second.get_value<string>();

                                    if(!xn.empty())
                                        currentImage.setXn(xn);
                                }

                                if(h.first == "yn")
                                {
                                    string yn = h.second.get_value<string>();

                                    if(!yn.empty())
                                        currentImage.setYn(yn);
                                }
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

vector<JSONImage> FileManager::GetTrainingSet(vector<string> *files)
{

}

vector<JSONImage> FileManager::GetTestSet(vector<string> *files)
{

}

vector<JSONImage> FileManager::GetValidationSet(vector<string> *files)
{

}
