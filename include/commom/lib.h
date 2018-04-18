//
// Created by markz on 2018-03-18.
//

#ifndef EFANNA2E_LIB_H
#define EFANNA2E_LIB_H
#include <random>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <cassert>
#include <omp.h>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <stdlib.h>

using namespace std;
extern unordered_map<string, time_t> timmer_;
extern unordered_map<string, string> record;
extern unordered_map<string, string> params;
#include <commom/util.h>
#include <commom/distance.h>
#include <commom/parameters.h>
#include <commom/neighbor.h>
#include <commom/exceptions.h>
#include <index/index.h>
#ifdef linux
#include<gperftools/profiler.h>
#endif
#ifdef __APPLE__
#else
#include <malloc.h>
#endif
using namespace efanna2e;

#endif //EFANNA2E_LIB_H
