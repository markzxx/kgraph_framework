//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#include <index/index_random.h>


namespace efanna2e {

IndexRandom::IndexRandom(const size_t dimension, const size_t n):Index(dimension, n, L2){
  has_built = true;
}
IndexRandom::~IndexRandom() {}
void IndexRandom::Build(size_t n, const float *data, const Parameters &parameters) {
  data_ = data;
    N = n;
    unsigned K = parameters.Get<unsigned>("K");
    final_graph_.resize(N);
#pragma omp parallel for
    for (unsigned i = 0; i < N; i++) {
        const float *query = data_ + i * dim_;
        std::vector<unsigned> tmp(K + 1);
        Search(query, data_, K + 1, parameters, tmp.data());

        for (unsigned j = 0; j < K; j++) {
            unsigned id = tmp[j];
            if (id == i)continue;
            final_graph_[i].push_back(id);
        }
    }
  // Do Nothing

  has_built = true;
}
void IndexRandom::Search(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices) {

    GenRandom(rng, indices, k, N);
}

}
