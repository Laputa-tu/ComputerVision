# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kevin/ComputerVision

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevin/ComputerVision

# Include any dependencies generated for this target.
include CMakeFiles/Main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Main.dir/flags.make

CMakeFiles/Main.dir/Main.cpp.o: CMakeFiles/Main.dir/flags.make
CMakeFiles/Main.dir/Main.cpp.o: Main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/kevin/ComputerVision/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Main.dir/Main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/Main.cpp.o -c /home/kevin/ComputerVision/Main.cpp

CMakeFiles/Main.dir/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/Main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/kevin/ComputerVision/Main.cpp > CMakeFiles/Main.dir/Main.cpp.i

CMakeFiles/Main.dir/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/Main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/kevin/ComputerVision/Main.cpp -o CMakeFiles/Main.dir/Main.cpp.s

CMakeFiles/Main.dir/Main.cpp.o.requires:
.PHONY : CMakeFiles/Main.dir/Main.cpp.o.requires

CMakeFiles/Main.dir/Main.cpp.o.provides: CMakeFiles/Main.dir/Main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Main.cpp.o.provides.build
.PHONY : CMakeFiles/Main.dir/Main.cpp.o.provides

CMakeFiles/Main.dir/Main.cpp.o.provides.build: CMakeFiles/Main.dir/Main.cpp.o

CMakeFiles/Main.dir/HOG/classifier.cc.o: CMakeFiles/Main.dir/flags.make
CMakeFiles/Main.dir/HOG/classifier.cc.o: HOG/classifier.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/kevin/ComputerVision/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Main.dir/HOG/classifier.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/HOG/classifier.cc.o -c /home/kevin/ComputerVision/HOG/classifier.cc

CMakeFiles/Main.dir/HOG/classifier.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/HOG/classifier.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/kevin/ComputerVision/HOG/classifier.cc > CMakeFiles/Main.dir/HOG/classifier.cc.i

CMakeFiles/Main.dir/HOG/classifier.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/HOG/classifier.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/kevin/ComputerVision/HOG/classifier.cc -o CMakeFiles/Main.dir/HOG/classifier.cc.s

CMakeFiles/Main.dir/HOG/classifier.cc.o.requires:
.PHONY : CMakeFiles/Main.dir/HOG/classifier.cc.o.requires

CMakeFiles/Main.dir/HOG/classifier.cc.o.provides: CMakeFiles/Main.dir/HOG/classifier.cc.o.requires
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/HOG/classifier.cc.o.provides.build
.PHONY : CMakeFiles/Main.dir/HOG/classifier.cc.o.provides

CMakeFiles/Main.dir/HOG/classifier.cc.o.provides.build: CMakeFiles/Main.dir/HOG/classifier.cc.o

CMakeFiles/Main.dir/HOG/clipper.cpp.o: CMakeFiles/Main.dir/flags.make
CMakeFiles/Main.dir/HOG/clipper.cpp.o: HOG/clipper.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/kevin/ComputerVision/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Main.dir/HOG/clipper.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/HOG/clipper.cpp.o -c /home/kevin/ComputerVision/HOG/clipper.cpp

CMakeFiles/Main.dir/HOG/clipper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/HOG/clipper.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/kevin/ComputerVision/HOG/clipper.cpp > CMakeFiles/Main.dir/HOG/clipper.cpp.i

CMakeFiles/Main.dir/HOG/clipper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/HOG/clipper.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/kevin/ComputerVision/HOG/clipper.cpp -o CMakeFiles/Main.dir/HOG/clipper.cpp.s

CMakeFiles/Main.dir/HOG/clipper.cpp.o.requires:
.PHONY : CMakeFiles/Main.dir/HOG/clipper.cpp.o.requires

CMakeFiles/Main.dir/HOG/clipper.cpp.o.provides: CMakeFiles/Main.dir/HOG/clipper.cpp.o.requires
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/HOG/clipper.cpp.o.provides.build
.PHONY : CMakeFiles/Main.dir/HOG/clipper.cpp.o.provides

CMakeFiles/Main.dir/HOG/clipper.cpp.o.provides.build: CMakeFiles/Main.dir/HOG/clipper.cpp.o

# Object files for target Main
Main_OBJECTS = \
"CMakeFiles/Main.dir/Main.cpp.o" \
"CMakeFiles/Main.dir/HOG/classifier.cc.o" \
"CMakeFiles/Main.dir/HOG/clipper.cpp.o"

# External object files for target Main
Main_EXTERNAL_OBJECTS =

Main: CMakeFiles/Main.dir/Main.cpp.o
Main: CMakeFiles/Main.dir/HOG/classifier.cc.o
Main: CMakeFiles/Main.dir/HOG/clipper.cpp.o
Main: CMakeFiles/Main.dir/build.make
Main: /home/kevin/opencv-2.4.9/lib/libopencv_videostab.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_video.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_ts.a
Main: /home/kevin/opencv-2.4.9/lib/libopencv_superres.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_stitching.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_photo.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_ocl.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_objdetect.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_nonfree.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_ml.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_legacy.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_imgproc.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_highgui.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_gpu.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_flann.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_features2d.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_core.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_contrib.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_calib3d.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_nonfree.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_ocl.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_gpu.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_photo.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_objdetect.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_legacy.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_video.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_ml.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_calib3d.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_features2d.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_highgui.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_imgproc.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_flann.so.2.4.9
Main: /home/kevin/opencv-2.4.9/lib/libopencv_core.so.2.4.9
Main: CMakeFiles/Main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Main.dir/build: Main
.PHONY : CMakeFiles/Main.dir/build

CMakeFiles/Main.dir/requires: CMakeFiles/Main.dir/Main.cpp.o.requires
CMakeFiles/Main.dir/requires: CMakeFiles/Main.dir/HOG/classifier.cc.o.requires
CMakeFiles/Main.dir/requires: CMakeFiles/Main.dir/HOG/clipper.cpp.o.requires
.PHONY : CMakeFiles/Main.dir/requires

CMakeFiles/Main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Main.dir/clean

CMakeFiles/Main.dir/depend:
	cd /home/kevin/ComputerVision && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevin/ComputerVision /home/kevin/ComputerVision /home/kevin/ComputerVision /home/kevin/ComputerVision /home/kevin/ComputerVision/CMakeFiles/Main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Main.dir/depend

