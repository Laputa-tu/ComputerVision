#include "../Classifier/classifier.h"
#include "./ImageObj.h"

using namespace std;

class JSONImage: public ImageObj
{
    protected:
        vector<double> xn;
        vector<double> yn;
    public:
        void setXn(string xn_string);
        void setYn(string yn_string);
        vector<double> getXn();
        vector<double> getYn();
        void printXn();
        void printYn();
        bool hasPolygon();
        ClipperLib::Path getLabelPolygon();

        static vector<double> StringToVector(string);
};
