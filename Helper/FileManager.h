#ifdef _MSC_VER
#include <boost/config/compiler/visualc.hpp>
#endif
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <cassert>
#include <exception>

#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <dirent.h>
#include <iostream>
#include <string>

#include "../Data/JSONImage.h"

//definitions
#define SLOTH_ZEBRA "zebra.json"
#define SLOTH_SIGN "sign.json"

using namespace std;

class FileManager
{
    private:
        
    public:
        FileManager(void);
        ~FileManager(void);

        static void GetFilesInDirectory(char *, const char*, int, vector<string> *);
        static void GetFilesInDirectory(char *, const char*, vector<string> *);
        static vector<JSONImage> GetJSONImages(vector<string> *);
};
