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
    parameter(argc, argv, params);
    float *data_load = NULL;
    unsigned *graph_truth = NULL;
    unsigned points_num, dim;
    unsigned points_num2, dim2;
    char data_file[50];
    char truth_file[50];
    string file = params.count("-f") ? params["-f"] : "siftsmall";
    sprintf(data_file, "data/%s/base.fvecs", file.c_str());
    addRecord("algorithm", argv[0]);
    sprintf(truth_file, "data/%s/graphtruth.ivecs", file.c_str());
    addRecord("file", file);
    load_data(data_file, data_load, points_num, dim);
    load_datai(truth_file, graph_truth, points_num2, dim2);

//    char* graph_filename = argv[3];
    unsigned iter = params.count("-i") ? stoi(params["-i"]) : 10;
    unsigned K = params.count("-k") ? stoi(params["-k"]) : 100;
    addRecord("K", to_string(K));
    unsigned L = params.count("-l") ? stoi(params["-l"]) : K;
    addRecord("L", to_string(L));
    unsigned S = params.count("-s") ? stoi(params["-s"]) : K;
    addRecord("S", to_string(S));
    unsigned R = params.count("-r") ? stoi(params["-r"]) : K;
    addRecord("R", to_string(R));
    string note = params.count("-note") ? params["-note"] : "";
    addRecord("note", note);
    string db = params.count("-db") ? params["-db"] : "y";
    addRecord("db", db);

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build

    efanna2e::IndexRandom init_index(dim, points_num);
    timmer("s_init");
    init_index.Build(points_num, data_load, paras);
    timmer("e_init");
    output_time("Init time", "s_init", "e_init");
    addRecord("init_time", dtos(timeby("s_init", "e_init"), 1));

    efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index *) (&init_index));
    index.SetGraph(init_index.GetGraph());
    index.SetGraphTruth(graph_truth, dim2);

    timmer("s_refine");
    index.RefineGraph(data_load, paras);
    output_time("Refine time", "s_refine", "e_refine");
    output_time("Total time", "s_init", "e_refine");
    printf("\n\n");

    return 0;
}

