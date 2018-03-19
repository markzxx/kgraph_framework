//
// Created by markz on 2018-03-18.
//

#ifndef EFANNA2E_INDEX_LSH_H
#define EFANNA2E_INDEX_LSH_H

#include <commom/lib.h>

namespace efanna2e {

    class IndexLSH : public Index {
    public:

        typedef std::vector<unsigned> Code;
        typedef std::vector<Code> Codes;

        IndexLSH(const size_t dimension, const size_t n, const float *data, Metric m, Parameters params);

        virtual ~IndexLSH();

        std::mt19937 rng;

        void Save(const char *filename) override {}

        void Load(const char *filename) override {}

        void Build();

        virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

        void Search(
                const float *query,
                const float *x,
                size_t k,
                const Parameters &parameters,
                unsigned *indices) override;

    protected:

        void random_projection(const float *data, size_t points_num, unsigned dim, unsigned codelen, Codes &output);

        void generate_random_projection_matrix(int dim, int codelen, float *projection_matrix);

        void init_graph();

        unsigned codelen_;
        unsigned numTable_;
        float *projection_matrix = NULL;

        Codes BaseCode;
        Codes QueryCode;

    };

}

#endif //EFANNA2E_INDEX_LSH_H
