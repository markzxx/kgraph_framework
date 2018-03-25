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

void load_datai(char *filename, unsigned *&data, unsigned &num, unsigned &dim) {// load graph by int
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        cout << "open file error" << endl;
        return;
    }
    in.read((char *) &dim, 4);
    in.seekg(0, ios::end);
    ios::pos_type ss = in.tellg();
    int fsize = (int) ss;
    num = fsize / (dim + 1) / 4;
    data = new unsigned[num * dim];

    in.seekg(0, ios::beg);
    for (unsigned i = 0; i < num; i++) {
        in.seekg(4, ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

int main(int argc, char **argv) {
    if (argc != 8) {
        std::cout << argv[0] << " data_file graph_truth iter L S R K" << std::endl;
        exit(-1);
    }
    float *data_load = NULL;
    unsigned *graph_truth = NULL;
    unsigned points_num, dim;
    unsigned points_num2, dim2;
    load_data(argv[1], data_load, points_num, dim);
    load_datai(argv[2], graph_truth, points_num2, dim2);

//    char* graph_filename = argv[3];
    unsigned iter = (unsigned) atoi(argv[3]);
    unsigned L = (unsigned) atoi(argv[4]);
    unsigned S = (unsigned) atoi(argv[5]);
    unsigned R = (unsigned) atoi(argv[6]);
    unsigned K = (unsigned) atoi(argv[7]);

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build

    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index *) (&init_index));
    index.SetGraphTruth(graph_truth, dim2);

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(points_num, data_load, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Refine time: " << diff.count() << "s\n";
    std::cout << "total time: " << diff.count() << "s\n";

    vector<std::vector<unsigned> > &final_result = index.GetGraph();

    int cnt = 0;
    for (unsigned i = 0; i < points_num2; i++) {
        for (unsigned j = 0; j < K; j++) {
            unsigned k = 0;
            for (; k < K; k++) {
                if (graph_truth[i * dim2 + j] == final_result[i][k]) break;
            }

            if (k == K)cnt++;
        }
    }
    float accuracy = 1 - (float) cnt / (points_num * K);
    cout << K << "NN accuracy: " << accuracy << endl;
    return 0;
}

