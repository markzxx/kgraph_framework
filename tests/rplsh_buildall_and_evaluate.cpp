//
// Created by markz on 2018-03-19.
//

//
// Created by markz on 2018-03-16.
//
#include <commom/lib.h>
#include <index/index_graph.h>
#include <index/index_random.h>
#include <index/index_lsh.h>

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
    if (argc != 11) {
        std::cout << argv[0] << " data_file graph_truth nTress mLevel iter L S R K" << std::endl;
        exit(-1);
    }
    float *data_load = NULL;
    unsigned *graph_truth = NULL;
    unsigned points_num, dim;
    unsigned points_num2, dim2;
    load_data(argv[1], data_load, points_num, dim);
    load_datai(argv[2], graph_truth, points_num2, dim2);


//    char* graph_filename = argv[3];
    unsigned numTable = (unsigned) atoi(argv[3]);
    unsigned codelen = (unsigned) atoi(argv[4]);
    unsigned iter = (unsigned) atoi(argv[5]);
    unsigned L = (unsigned) atoi(argv[6]);
    unsigned S = (unsigned) atoi(argv[7]);
    unsigned R = (unsigned) atoi(argv[8]);
    unsigned K = (unsigned) atoi(argv[9]);
    unsigned threads = (unsigned) atoi(argv[10]);

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("numTable", numTable);
    paras.Set<unsigned>("codelen", codelen);

    data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
    efanna2e::IndexLSH init_index(dim, points_num, data_load, efanna2e::L2, paras);

    timmer("s_init");
    init_index.Build();
    timmer("e_init");
    printf("Init time:%.1f\n", timeby("s_init", "e_init"));

    efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index *) (&init_index));
    index.SetGraph(init_index.GetGraph()); //pass the init graph without Save and Load
    index.SetGraphTruth(graph_truth, dim2);

    timmer("s_refine");
    index.RefineGraph(data_load, paras);
    timmer("e_refine");
    printf("Refine time:%.1f\n", timeby("s_refine", "e_refine"));
    printf("Total time:%.1f\n", timeby("s_init", "e_refine"));

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
