#include "JSONImage.h"

void ImageObj::setName(string name)
{
    this->name = name;
}
string ImageObj::getName()
{
    return this->name;
}

void ImageObj::setPath(string path)
{
    this->path = path;
}
string ImageObj::getPath()
{
    return this->path;
}
