# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.17.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.17.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/pu/Downloads/libjpeg-turbo-2.0.3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/pu/Downloads/libjpeg-turbo-2.0.3/build

# Utility rule file for testclean.

# Include the progress variables for this target.
include CMakeFiles/testclean.dir/progress.make

CMakeFiles/testclean:
	/usr/local/Cellar/cmake/3.17.3/bin/cmake -P /Users/pu/Downloads/libjpeg-turbo-2.0.3/cmakescripts/testclean.cmake

testclean: CMakeFiles/testclean
testclean: CMakeFiles/testclean.dir/build.make

.PHONY : testclean

# Rule to build all files generated by this target.
CMakeFiles/testclean.dir/build: testclean

.PHONY : CMakeFiles/testclean.dir/build

CMakeFiles/testclean.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testclean.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testclean.dir/clean

CMakeFiles/testclean.dir/depend:
	cd /Users/pu/Downloads/libjpeg-turbo-2.0.3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/pu/Downloads/libjpeg-turbo-2.0.3 /Users/pu/Downloads/libjpeg-turbo-2.0.3 /Users/pu/Downloads/libjpeg-turbo-2.0.3/build /Users/pu/Downloads/libjpeg-turbo-2.0.3/build /Users/pu/Downloads/libjpeg-turbo-2.0.3/build/CMakeFiles/testclean.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testclean.dir/depend
