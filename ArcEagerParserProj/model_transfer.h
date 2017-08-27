#pragma once
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "cnn/model.h"


using namespace cnn;
using namespace std;

void transfer_args(parameters_store& ps, Model& mdl, const char* inifile);

void load_string2idx(const char* path, map<string, unsigned>& locvar2orgvar);
void save_string2idx(const char* path, map<string, unsigned>& locvar2orgvar);

void convert_args(parameters_store& ps, Model& mdl, const char* inifile, map<string, unsigned>& org_word2idx, map<string, unsigned>& cur_word2idx);
