#include "FileManager.h"

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

vector<JSONImage> FileManager::GetJSONImages(vector<string> *files)
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
