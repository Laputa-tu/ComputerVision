# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running interactive CMake command-line interface..."
	/usr/bin/cmake -i .
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/kevin/ComputerVision/CMakeFiles /home/kevin/ComputerVision/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/kevin/ComputerVision/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named Main

# Build rule for target.
Main: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 Main
.PHONY : Main

# fast build rule for target.
Main/fast:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/build
.PHONY : Main/fast

Classifier/classifier.o: Classifier/classifier.cc.o
.PHONY : Classifier/classifier.o

# target to build an object file
Classifier/classifier.cc.o:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Classifier/classifier.cc.o
.PHONY : Classifier/classifier.cc.o

Classifier/classifier.i: Classifier/classifier.cc.i
.PHONY : Classifier/classifier.i

# target to preprocess a source file
Classifier/classifier.cc.i:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Classifier/classifier.cc.i
.PHONY : Classifier/classifier.cc.i

Classifier/classifier.s: Classifier/classifier.cc.s
.PHONY : Classifier/classifier.s

# target to generate assembly for a file
Classifier/classifier.cc.s:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Classifier/classifier.cc.s
.PHONY : Classifier/classifier.cc.s

Classifier/clipper.o: Classifier/clipper.cpp.o
.PHONY : Classifier/clipper.o

# target to build an object file
Classifier/clipper.cpp.o:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Classifier/clipper.cpp.o
.PHONY : Classifier/clipper.cpp.o

Classifier/clipper.i: Classifier/clipper.cpp.i
.PHONY : Classifier/clipper.i

# target to preprocess a source file
Classifier/clipper.cpp.i:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Classifier/clipper.cpp.i
.PHONY : Classifier/clipper.cpp.i

Classifier/clipper.s: Classifier/clipper.cpp.s
.PHONY : Classifier/clipper.s

# target to generate assembly for a file
Classifier/clipper.cpp.s:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Classifier/clipper.cpp.s
.PHONY : Classifier/clipper.cpp.s

Data/JSONImage.o: Data/JSONImage.cpp.o
.PHONY : Data/JSONImage.o

# target to build an object file
Data/JSONImage.cpp.o:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Data/JSONImage.cpp.o
.PHONY : Data/JSONImage.cpp.o

Data/JSONImage.i: Data/JSONImage.cpp.i
.PHONY : Data/JSONImage.i

# target to preprocess a source file
Data/JSONImage.cpp.i:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Data/JSONImage.cpp.i
.PHONY : Data/JSONImage.cpp.i

Data/JSONImage.s: Data/JSONImage.cpp.s
.PHONY : Data/JSONImage.s

# target to generate assembly for a file
Data/JSONImage.cpp.s:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Data/JSONImage.cpp.s
.PHONY : Data/JSONImage.cpp.s

Helper/FileManager.o: Helper/FileManager.cpp.o
.PHONY : Helper/FileManager.o

# target to build an object file
Helper/FileManager.cpp.o:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Helper/FileManager.cpp.o
.PHONY : Helper/FileManager.cpp.o

Helper/FileManager.i: Helper/FileManager.cpp.i
.PHONY : Helper/FileManager.i

# target to preprocess a source file
Helper/FileManager.cpp.i:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Helper/FileManager.cpp.i
.PHONY : Helper/FileManager.cpp.i

Helper/FileManager.s: Helper/FileManager.cpp.s
.PHONY : Helper/FileManager.s

# target to generate assembly for a file
Helper/FileManager.cpp.s:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Helper/FileManager.cpp.s
.PHONY : Helper/FileManager.cpp.s

LBP/LBP.o: LBP/LBP.cpp.o
.PHONY : LBP/LBP.o

# target to build an object file
LBP/LBP.cpp.o:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/LBP/LBP.cpp.o
.PHONY : LBP/LBP.cpp.o

LBP/LBP.i: LBP/LBP.cpp.i
.PHONY : LBP/LBP.i

# target to preprocess a source file
LBP/LBP.cpp.i:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/LBP/LBP.cpp.i
.PHONY : LBP/LBP.cpp.i

LBP/LBP.s: LBP/LBP.cpp.s
.PHONY : LBP/LBP.s

# target to generate assembly for a file
LBP/LBP.cpp.s:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/LBP/LBP.cpp.s
.PHONY : LBP/LBP.cpp.s

Main.o: Main.cpp.o
.PHONY : Main.o

# target to build an object file
Main.cpp.o:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Main.cpp.o
.PHONY : Main.cpp.o

Main.i: Main.cpp.i
.PHONY : Main.i

# target to preprocess a source file
Main.cpp.i:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Main.cpp.i
.PHONY : Main.cpp.i

Main.s: Main.cpp.s
.PHONY : Main.s

# target to generate assembly for a file
Main.cpp.s:
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/Main.cpp.s
.PHONY : Main.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... Main"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... Classifier/classifier.o"
	@echo "... Classifier/classifier.i"
	@echo "... Classifier/classifier.s"
	@echo "... Classifier/clipper.o"
	@echo "... Classifier/clipper.i"
	@echo "... Classifier/clipper.s"
	@echo "... Data/JSONImage.o"
	@echo "... Data/JSONImage.i"
	@echo "... Data/JSONImage.s"
	@echo "... Helper/FileManager.o"
	@echo "... Helper/FileManager.i"
	@echo "... Helper/FileManager.s"
	@echo "... LBP/LBP.o"
	@echo "... LBP/LBP.i"
	@echo "... LBP/LBP.s"
	@echo "... Main.o"
	@echo "... Main.i"
	@echo "... Main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

