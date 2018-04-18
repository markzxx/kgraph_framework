//
// Created by markz on 2018-03-21.
//
//
// Created by markz on 2018-03-16.
//

#include <commom/lib.h>
#include <index/index_graph.h>
#include <index/index_random.h>
#include <index/index_kdtree.h>


void load_data(char *filename, float *&data, unsigned &num, unsigned &dim) {// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

struct neighbour {
    unsigned id;
    float dist;

    neighbour(const unsigned id, const float dist) : id(id), dist(dist) {}

    bool operator<(const neighbour &rhs) const {
        if (this->id == rhs.id)
            return false;
        if (this->dist == rhs.dist) {
            return this->id > rhs.id;
        }
        return this->dist < rhs.dist;
    }
};

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << argv[0] << " data_file K" << std::endl;
        exit(-1);
    }
    float *data_load = NULL;
    unsigned points_num, dim;
    char *data_file = new char[50];
    sprintf(data_file, "data/%s/base.fvecs", argv[1]);
    load_data(data_file, data_load, points_num, dim);
    unsigned K = stoi(argv[2]);
    vector<vector<neighbour>> NNheap;
    NNheap.resize(points_num);
    DistanceL2 L2 = DistanceL2();
    timmer("begin");
    for (unsigned i = 0; i < points_num; i++) {
        auto &nni = NNheap[i];
        for (unsigned j = i + 1; j < points_num; j++) {
            auto &nnj = NNheap[j];
            float dist = L2.compare(data_load + dim * i, data_load + dim * j, dim);

            if (nni.size() < K) {
                nni.emplace_back(neighbour(j, dist));
                push_heap(nni.begin(), nni.end());
            } else if (dist < nni.front().dist) {
                pop_heap(nni.begin(), nni.end());
                nni.back() = neighbour(j, dist);
                push_heap(nni.begin(), nni.end());
            }

            if (nnj.size() < K) {
                nnj.emplace_back(neighbour(i, dist));
                push_heap(nnj.begin(), nnj.end());
            } else if (dist < nnj.front().dist) {
                pop_heap(nnj.begin(), nnj.end());
                nnj.back() = neighbour(i, dist);
                push_heap(nnj.begin(), nnj.end());
            }
        }
    }
    timmer("end");
    output_time("Brute Force time", "begin", "end");
    return 0;
}

