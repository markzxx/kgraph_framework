# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2017.2.2\bin\cmake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2017.2.2\bin\cmake\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug

# Include any dependencies generated for this target.
include tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/flags.make

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/flags.make
tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/includes_CXX.rsp
tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj: ../tests/kdtree_buildall_and_evaluate.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj"
	cd /d C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\tests && C:\MinGW\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\kdtree_buildall_and_evaluate.dir\kdtree_buildall_and_evaluate.cpp.obj -c C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\tests\kdtree_buildall_and_evaluate.cpp

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.i"
	cd /d C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\tests && C:\MinGW\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\tests\kdtree_buildall_and_evaluate.cpp > CMakeFiles\kdtree_buildall_and_evaluate.dir\kdtree_buildall_and_evaluate.cpp.i

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.s"
	cd /d C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\tests && C:\MinGW\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\tests\kdtree_buildall_and_evaluate.cpp -o CMakeFiles\kdtree_buildall_and_evaluate.dir\kdtree_buildall_and_evaluate.cpp.s

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj.requires:

.PHONY : tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj.requires

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj.provides: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj.requires
	$(MAKE) -f tests\CMakeFiles\kdtree_buildall_and_evaluate.dir\build.make tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj.provides.build
.PHONY : tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj.provides

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj.provides.build: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj


# Object files for target kdtree_buildall_and_evaluate
kdtree_buildall_and_evaluate_OBJECTS = \
"CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj"

# External object files for target kdtree_buildall_and_evaluate
kdtree_buildall_and_evaluate_EXTERNAL_OBJECTS =

tests/kdtree_buildall_and_evaluate.exe: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj
tests/kdtree_buildall_and_evaluate.exe: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/build.make
tests/kdtree_buildall_and_evaluate.exe: src/libefanna2e.a
tests/kdtree_buildall_and_evaluate.exe: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/linklibs.rsp
tests/kdtree_buildall_and_evaluate.exe: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/objects1.rsp
tests/kdtree_buildall_and_evaluate.exe: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable kdtree_buildall_and_evaluate.exe"
	cd /d C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\kdtree_buildall_and_evaluate.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/build: tests/kdtree_buildall_and_evaluate.exe

.PHONY : tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/build

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/requires: tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/kdtree_buildall_and_evaluate.cpp.obj.requires

.PHONY : tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/requires

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/clean:
	cd /d C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\tests && $(CMAKE_COMMAND) -P CMakeFiles\kdtree_buildall_and_evaluate.dir\cmake_clean.cmake
.PHONY : tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/clean

tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\tests C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\tests C:\Users\markz\Desktop\KNN-Graph\knngFramework\kgraph_framework\cmake-build-debug\tests\CMakeFiles\kdtree_buildall_and_evaluate.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/kdtree_buildall_and_evaluate.dir/depend
