cmake_minimum_required(VERSION 2.8)
project( Main )
find_package( OpenCV REQUIRED )

# add vlfeat library
add_library(vlfeat_slic lib_vlfeat/vl/host.c
lib_vlfeat/vl/random.c
lib_vlfeat/vl/generic.c
lib_vlfeat/vl/lbp.c)

add_executable( Main Main.h  Main.cpp Classifier/classifier.h Classifier/classifier.cc Classifier/clipper.hpp Classifier/clipper.cpp Data/ImageObj.h Data/ImageObj.cpp Data/JSONImage.h Data/JSONImage.cpp Helper/FileManager.h Helper/FileManager.cpp LBP/lbpfeature.cpp LBP/lbp.hpp LBP/lbp.cpp LBP/histogram.cpp LBP/histogram.hpp)
target_link_libraries( Main ${OpenCV_LIBS} vlfeat_slic -lpthread -std=gnu++11)


