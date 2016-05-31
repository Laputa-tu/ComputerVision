#include "../HOG/classifier.h"

class JSONImage
{
    private:
        string name, path;
        vector<double> xn;
        vector<double> yn;
    public:
        JSONImage(void);
        ~JSONImage(void);

        void setXn(string xn_string);
        void setYn(string yn_string);
        vector<double> getXn();
        vector<double> getYn();
        void printXn();
        void printYn();
        void setName(string name);
        string getName();
        void setPath(string path);
        string getPath();
        bool hasPolygon();
        ClipperLib::Path getLabelPolygon();

        static vector<double> StringToVector(string);
};
