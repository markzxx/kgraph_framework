//
// Created by markz on 2018-03-18.
//

#ifndef EFANNA2E_INDEX_LSH_H
#define EFANNA2E_INDEX_LSH_H

#include <commom/lib.h>

struct Candidate2 {
    unsigned id;
    float distance;

    Candidate2(const unsigned row_id, const float distance) : id(row_id), distance(distance) {}

    bool operator>(const Candidate2 &rhs) const {
        if (this->id == rhs.id)
            return false;
        if (this->distance == rhs.distance) {
            return this->id > rhs.id;
        }
        return this->distance > rhs.distance;
    }

    bool operator<(const Candidate2 &rhs) const {
        if (this->id == rhs.id)
            return false;
        if (this->distance == rhs.distance) {
            return this->id < rhs.id;
        }
        return this->distance < rhs.distance;
    }
};

struct Code {
    unsigned len;
    long long code;

    Code(unsigned l, long long c) : len(l), code(c) {}

    bool operator==(const Code &other) const {
        return this->code == other.code && this->len == other.code;
    }
};

struct mapHashFunc {
    std::size_t operator()(const Code &key) const {
        using std::size_t;
        using std::hash;

        return ((hash<unsigned>()(key.len) ^ (hash<long long>()(key.code) << 1)) >> 1);
    }
};

namespace efanna2e {

    class IndexLSH : public Index {
    public:

        typedef std::vector<unsigned> Code2;
        typedef std::vector<Code2> Codes2;
        typedef unordered_multimap<unsigned, unsigned> Bucket;
        typedef vector<Bucket> HashTable;
        typedef vector<set<Candidate2>> CandidateHeap2;

        typedef float *HashFunc;
        typedef vector<HashFunc> HashFamily;
        typedef vector<HashFamily> HashFamilys;
        typedef vector<Code> Codes;
        typedef vector<Codes> CodeFamilys;
        typedef unordered_multimap<Code, unsigned, mapHashFunc> CLSH_HashTable;
        typedef vector<CLSH_HashTable> CLSH_HashTables;

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

        void random_projection(const float *data, size_t points_num, unsigned dim, unsigned codelen, Codes2 &output);

        void generate_random_projection_matrix(int dim, int codelen, float *projection_matrix);

        void buildHashTable();

        void build_CLSH_HashTable();

        void CLSH_init();

        void CLSH(unsigned *ids, unsigned *code, unsigned left, unsigned right, unsigned famid, unsigned len);

        void init_graph();

        void CLSH_bucketIndex();

        void bucketIndex();

        void navie_bucketIndex();

        void navie_bucketIndex2();

        void vec_navie_bucketIndex();

        inline void clearHashtable() {
#pragma omp parallel for
            for (int i = 0; i < numTable_; i++) {
                Code2 &baseCode = BaseCode[i];
                Code2().swap(baseCode);
                Bucket &bucket = hashtable[i];
                bucket.clear();
            }
        }

        inline void clearCLSHHashtable() {
#pragma omp parallel for
            for (int i = 0; i < numTable_; i++) {
                auto &table = clsh_hashTables[i];
                unordered_multimap<Code, unsigned, mapHashFunc>().swap(table);
            }
            clsh_hashTables.clear();
        }

        void extendHashFamily(unsigned int famid);

        unsigned codelen_;
        unsigned tablelen_;
        unsigned numTable_;
        unsigned K_;
        float *projection_matrix = NULL;


        HashFamilys hashFamilys;
        CodeFamilys codeFamilys;
        CLSH_HashTables clsh_hashTables;
        unsigned maxlen = 0;
        vector<unordered_map<unsigned, unsigned >> clshBucket;
        DistanceInnerProduct dist;
        long long build_com = 0;

        CandidateHeap2 knn_graph;
        Codes2 BaseCode;
        Codes2 QueryCode;
        HashTable hashtable;

    };

}

#endif //EFANNA2E_INDEX_LSH_H
