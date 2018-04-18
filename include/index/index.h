//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_INDEX_H
#define EFANNA2E_INDEX_H

#include <commom/lib.h>

namespace efanna2e {

    class Index {
    public:
        typedef std::vector<std::vector<unsigned> > CompactGraph;

        explicit Index(const size_t dimension, const size_t n, Metric metric);

        virtual ~Index();

        virtual void Build(size_t n, const float *data, const Parameters &parameters) = 0;

        virtual void Search(
                const float *query,
                const float *x,
                size_t k,
                const Parameters &parameters,
                unsigned *indices) = 0;

        virtual void Save(const char *filename) = 0;

        virtual void Load(const char *filename) = 0;

        inline bool HasBuilt() const { return has_built; }

        inline size_t GetDimension() const { return dim_; };

        inline size_t GetSizeOfDataset() const { return N; }

        inline const float *GetDataset() const { return data_; }

        inline CompactGraph &GetGraph() { return final_graph_; }

        inline void SetGraph(CompactGraph &graph) { final_graph_.swap(graph); }

        inline void SetGraphTruth(unsigned *truth, unsigned num) {
            graph_truth = truth;
            truthNum = num;
        }

    protected:
        unsigned dim_;
        const float *data_;
        unsigned N;
        bool has_built;
        Distance *distance_;
        Parameters params_;
        CompactGraph final_graph_;
        const unsigned *graph_truth;
        unsigned truthNum;
        long long nn_comp = 0;
    };

}

#endif //EFANNA2E_INDEX_H
