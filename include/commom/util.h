//
// Created by 付聪 on 2017/6/21.
//

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H

#include <commom/lib.h>
#include <commom/MyDB.h>
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

    inline void DBexec() {
        MyDB db;
        time_t date = time(0);
        char tmpBuf[255];
        strftime(tmpBuf, 255, "%Y%m%d%H%M", localtime(&date));
        record["date"] = tmpBuf;
        db.initDB("120.24.163.35", "mark", "123456", "experiment");
        db.addRecord("KNNG_mark", record);
    }

    inline void getSkyline(vector<vector<string>> &skyline) {
        MyDB db;
        db.initDB("120.24.163.35", "mark", "123456", "experiment");
        string sql =
                "select total_recall, total_time from knng_mark_skyline where K=" + record["K"] + " and algorithm=\"" +
                record["algorithm"] + "\" and file=\"" + record["file"] + "\"";
        if (!db.getData(sql, skyline)) {
            cout << "get skyline failed" << endl;
        };
        printf("skyline num:%d\n", skyline.size());
    }

    inline void addRecord(string key, string value) {
        record.operator[](key) = value;
    }

    inline string dtos(double d, int len) {
        char str[20];
        int tem = (int) pow(10, len);
        d *= tem;
        sprintf(str, "%d.%d", (int) d / tem, (int) d % tem);
        return str;
    }

    inline bool dominate(double recall, double time, vector<vector<string>> skyline) {
        for (auto &row : skyline) {
            if (stod(dtos(recall, 4)) <= stod(row[0]) && time >= stod(row[1])) {
                printf("dominate recall:%s\tdominate time:%s\n", row[0].c_str(), row[1].c_str());
                return true;
            }
        }
        return false;
    }

    inline void timmer(string s) {
        timmer_.operator[](s) = clock();
    }

    inline double timeby(string s, string e) {
        return double(timmer_[e] - timmer_[s]) / CLOCKS_PER_SEC;
    }

    inline void output_time(string content, string s, string e) {
        printf("%s:%ss\n", content.c_str(), dtos(timeby(s, e), 1).c_str());
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
