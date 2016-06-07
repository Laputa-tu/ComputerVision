#include "JSONImage.h"

void JSONImage::setXn(string xn_string)
{
    this->xn = StringToVector(xn_string);
}

void JSONImage::setYn(string yn_string)
{
    this->yn = StringToVector(yn_string);
}

vector<double> JSONImage::getXn()
{
    return this->xn;
}

vector<double> JSONImage::getYn()
{
    return this->yn;
}

void JSONImage::printXn()
{
    for(int i=0; i<xn.size(); i++)
    {
        cout.precision(17);
        cout << xn.at(i) << "; ";
    }

    cout << endl;
}

void JSONImage::printYn()
{
    for(int i=0; i<yn.size(); i++)
    {
        cout.precision(17);
        cout << yn.at(i) << "; ";
    }

    cout << endl;
}

bool JSONImage::hasPolygon()
{
    if(xn.size() > 0 && yn.size() > 0)
        return true;
    else
        return false;
}

ClipperLib::Path JSONImage::getLabelPolygon()
{
    ClipperLib::Path labelPolygon;
    for(int i=0; i<xn.size(); i++)
    {
        labelPolygon << ClipperLib::IntPoint(xn.at(i), yn.at(i));
    }
    return labelPolygon;
}

vector<double> JSONImage::StringToVector(string str)
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
