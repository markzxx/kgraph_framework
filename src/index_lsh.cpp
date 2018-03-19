//
// Created by markz on 2018-03-18.
//

#include <index/index_lsh.h>
#include <cblas.h>

namespace efanna2e {

    IndexLSH::IndexLSH(const size_t dimension, const size_t n, const float *data, Metric m, Parameters params) : Index(
            dimension, n, m) {
        data_ = data;
        params_ = params;
        codelen_ = params_.Get<unsigned>("codelen");
        numTable_ = params_.Get<unsigned>("numTable");
    }

    IndexLSH::~IndexLSH() {}

    void IndexLSH::Build() {

        projection_matrix = new float[codelen_ * dim_];

        generate_random_projection_matrix(dim_, codelen_, projection_matrix);
        random_projection(data_, N, dim_, codelen_, BaseCode);
        init_graph();

        has_built = true;
    }

    void
    IndexLSH::Search(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices) {

        GenRandom(rng, indices, k, N);
    }

    inline unsigned parallel_popcnt32(unsigned x) {
        x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f);
        x = (x & 0x00ff00ff) + ((x >> 8) & 0x00ff00ff);
        x = (x & 0x0000ffff) + ((x >> 16) & 0x0000ffff);
        return x;
    }

    void IndexLSH::init_graph() {
        unsigned L = params_.Get<unsigned>("L");
        unsigned K = params_.Get<unsigned>("K");
        for (size_t cur = 0; cur < N; cur++) {
            std::vector<int> hammingDistance(N);

            for (size_t j = 0; j < numTable_; j++) {
                for (size_t i = 0; i < N; i++) {
                    hammingDistance[i] += parallel_popcnt32(BaseCode[j][i] ^ QueryCode[j][cur]);
                }
            }

            // initialize index
            std::vector<size_t> hammingDistanceIndex(N);
            for (size_t i = 0; i < hammingDistanceIndex.size(); i++) {
                hammingDistanceIndex[i] = i;
            }
            // compare function
            auto compare_func = [&hammingDistance](const size_t a, const size_t b) -> bool {
                return hammingDistance[a] < hammingDistance[b];
            };
            // sort the index by its value
            std::partial_sort(hammingDistanceIndex.begin(), hammingDistanceIndex.begin() + L,
                              hammingDistanceIndex.end(), compare_func);

            std::vector<std::pair<float, size_t>> Distance;
            for (size_t i = 0; i < L; i++) {
                Distance.push_back(std::make_pair(
                        distance_->compare(data_ + cur * dim_, data_ + hammingDistanceIndex[i] * dim_, dim_),
                        hammingDistanceIndex[i]));
            }
            std::partial_sort(Distance.begin(), Distance.begin() + K, Distance.end());

            std::vector<unsigned> res;
            for (unsigned int j = 0; j < K; j++) res.push_back(Distance[j].second);
            final_graph_.push_back(res);
        }
    }

    void IndexLSH::generate_random_projection_matrix(int dim, int codelen, float *projection_matrix) {
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0, 1.0);
        for (int i = 0; i < codelen * dim; i++) {
            projection_matrix[i] = distribution(generator);
        }
    }

    void
    IndexLSH::random_projection(const float *data, size_t points_num, unsigned dim, unsigned codelen, Codes &output) {
        float *projection_result = new float[codelen * points_num];
        cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, points_num, codelen, dim,
                    1, data, dim, projection_matrix, codelen, 0, projection_result, points_num);

        output.resize(numTable_);
        for (unsigned i = 0; i < numTable_; i++) {
            output[i].resize(points_num);
            std::fill(output[i].begin(), output[i].end(), 0);
        }

        for (unsigned i = 0; i < codelen; i++) {
            const int table_id = i / 32;
            const unsigned bit = 1u << (i % 32);
            for (size_t j = 0; j < points_num; j++) {
                if (projection_result[i * points_num + j] > 0)
                    output[table_id][j] |= bit;
            }
        }

        delete[]projection_result;
    }

}