//
// Created by 付聪 on 2017/6/21.
//

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H

#include <commom/lib.h>

using namespace std;
namespace efanna2e {

    static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        set<unsigned> unique;
        while (unique.size() < size) {
            unique.insert(rng() % N);
        }
        auto it = unique.begin();
        for (unsigned i = 0; i < size; ++i, ++it) {
            addr[i] = *it;
        }
    }

    inline void parameter(int argc, char **argv, unordered_map<string, string> &map) {
        for (int i = 1; i < argc; i += 2) {
            map.insert({argv[i], argv[i + 1]});
        }
    }

    inline void addRecord(string key, string value) {
        record.insert({key, value});
    }

    inline void timmer(string s) {
        timmer_.insert({s, clock()});
    }

    inline string timeby(string s, string e) {
        char str[20];
        sprintf(str, "%.1f", double(timmer_[e] - timmer_[s]) / CLOCKS_PER_SEC);
        return str;
    }

    inline void output_time(string content, string s, string e) {
        printf("%s:%ss\n", content.c_str(), timeby(s, e).c_str());
    }

    inline float* data_align(float* data_ori, unsigned point_num, unsigned& dim){
      #ifdef __GNUC__
      #ifdef __AVX__
        #define DATA_ALIGN_FACTOR 8
      #else
      #ifdef __SSE2__
        #define DATA_ALIGN_FACTOR 4
      #else
        #define DATA_ALIGN_FACTOR 1
      #endif
      #endif
      #endif

//      std::cout << "align with : "<<DATA_ALIGN_FACTOR << std::endl;
      float* data_new=0;
      unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
//      std::cout << "align to new dim: "<<new_dim << std::endl;
      #ifdef __APPLE__
        data_new = new float[new_dim * point_num];
      #else
        data_new = (float*)malloc(point_num * new_dim * sizeof(float));
      #endif

      for(unsigned i=0; i<point_num; i++){
        memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
        memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
      }
      dim = new_dim;
      #ifdef __APPLE__
        delete[] data_ori;
      #else
        free(data_ori);
      #endif
      return data_new;
    }
}

#endif //EFANNA2E_UTIL_H
