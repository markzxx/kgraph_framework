//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//
#include <index/index.h>

unordered_map<string, time_t> timmer_;
unordered_map<string, string> record;
unordered_map<string, string> params;

namespace efanna2e {
    Index::Index(const size_t dimension, const size_t n, Metric metric = L2)
            : dim_(dimension), N(n), has_built(false) {
        switch (metric) {
            case L2:
                distance_ = new DistanceL2();
                break;
            default:
                distance_ = new DistanceL2();
                break;
        }
    }

    Index::~Index() {}
}
