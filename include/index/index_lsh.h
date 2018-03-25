//
// Created by markz on 2018-03-18.
//

#ifndef EFANNA2E_INDEX_LSH_H
#define EFANNA2E_INDEX_LSH_H

#include <commom/lib.h>

struct Candidate1 {
    size_t id;
    float distance;

    Candidate1(const size_t row_id, const float distance) : id(row_id), distance(distance) {}

    bool operator>(const Candidate1 &rhs) const {
        if (this->id == rhs.id)
            return false;
        if (this->distance == rhs.distance) {
            return this->id > rhs.id;
        }
        return this->distance > rhs.distance;
    }

    bool operator<(const Candidate1 &rhs) const {
        if (this->id == rhs.id)
            return false;
        if (this->distance == rhs.distance) {
            return this->id < rhs.id;
        }
        return this->distance < rhs.distance;
    }
};

namespace efanna2e {

    class IndexLSH : public Index {
    public:

        typedef std::vector<unsigned> Code;
        typedef std::vector<Code> Codes;
        typedef unordered_multimap<unsigned, unsigned> Bucket;
        typedef vector<Bucket> HashTable;
        typedef vector<set<Candidate1, greater<Candidate1>>> CandidateHeap1;


        IndexLSH(const size_t dimension, const size_t n, const float *data, Metric m, Parameters params);

        virtual ~IndexLSH();

        std::mt19937 rng;

        void Save(const char *filename) override {}

        void Load(const char *filename) override {}

        void Build();

        void Build(size_t n, const float *data, const Parameters &parameters) override;

        void Search(
                const float *query,
                const float *x,
                size_t k,
                const Parameters &parameters,
                unsigned *indices) override;

    protected:

        void random_projection(const float *data, size_t points_num, unsigned dim, unsigned codelen, Codes &output);

        void generate_random_projection_matrix(int dim, int codelen, float *projection_matrix);

        void buildHashTable();

        void init_graph();

        void bucketIndex();

        void navie_bucketIndex();

        void navie_bucketIndex2();

        inline void clearHashtable() {
#pragma omp parallel for
            for (int i = 0; i < numTable_; i++) {
                Code &baseCode = BaseCode[i];
                Code().swap(baseCode);
                Bucket &bucket = hashtable[i];
                bucket.clear();
            }
        }

        unsigned codelen_;
        unsigned tablelen_;
        unsigned numTable_;
        float *projection_matrix = NULL;

        CandidateHeap1 knn_graph;
        Codes BaseCode;
        Codes QueryCode;
        HashTable hashtable;

    };

}

#endif //EFANNA2E_INDEX_LSH_H
