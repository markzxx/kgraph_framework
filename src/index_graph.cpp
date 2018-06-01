//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#include <index/index_graph.h>
#include <commom/MyDB.h>
namespace efanna2e {
#define _CONTROL_NUM 1000

    IndexGraph::IndexGraph(const size_t dimension, const size_t n, Metric m, Index *initializer)
            : Index(dimension, n, m),
              initializer_{initializer} {
        assert(dimension == initializer->GetDimension());
    }

    IndexGraph::~IndexGraph() {}

    void IndexGraph::join() {
        unsigned L = graph_[0].L;
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (unsigned n = 0; n < N; n++) {
            graph_[n].join([&](unsigned i, unsigned j) {
                if (i != j) {
//                    if (!inSameBucket(inBuckets[i],inBuckets[j])) {
                    auto &nhoodi = graph_[i];
                    auto &nhoodj = graph_[j];
                    float dist = distance_->compare(data_ + i * dim_, data_ + j * dim_, dim_);
                    nn_comp++;
                    if (dist < nhoodi.pool.rbegin()->distance) {
                        nhoodi.pool.insert(Neighbor(j, dist, true));
                        if (nhoodi.pool.size() > L)
                            nhoodi.pool.erase(++nhoodi.pool.end());
                    }
                    if (dist < nhoodj.pool.rbegin()->distance) {
                        nhoodj.pool.insert(Neighbor(i, dist, true));
                        if (nhoodj.pool.size() > L)
                            nhoodj.pool.erase(++nhoodj.pool.end());
                    }
//                    }
                }
            });
        }
    }

    void IndexGraph::update(const Parameters &parameters) {
        unsigned S = parameters.Get<unsigned>("S");
        unsigned R = parameters.Get<unsigned>("R");
//        unsigned L = parameters.Get<unsigned>("L");
#pragma omp parallel for
        for (unsigned i = 0; i < N; i++) {
            std::vector<unsigned>().swap(graph_[i].nn_new);
            std::vector<unsigned>().swap(graph_[i].nn_old);
            //std::vector<unsigned>().swap(graph_[i].rnn_new);
            //std::vector<unsigned>().swap(graph_[i].rnn_old);
            //graph_[i].nn_new.clear();
            //graph_[i].nn_old.clear();
            //graph_[i].rnn_new.clear();
            //graph_[i].rnn_old.clear();
        }
#pragma omp parallel for
        for (unsigned n = 0; n < N; ++n) {
            auto &nn = graph_[n];
            unsigned maxl = std::min(nn.M + S, (unsigned) nn.pool.size());
            unsigned c = 0;
            unsigned l = 0;
            //std::sort(nn.pool.begin(), nn.pool.end());
            //if(n==0)std::cout << nn.pool[0].distance<<","<< nn.pool[1].distance<<","<< nn.pool[2].distance<< std::endl;
            auto it = nn.pool.begin();
            while ((l < maxl) && (c < S)) {
                if (it->flag) ++c;
                ++l;
                ++it;
            }
            nn.M = l;
        }
#pragma omp parallel for
        for (unsigned n = 0; n < N; ++n) {
            auto &nnhd = graph_[n];
            auto &nn_new = nnhd.nn_new;
            auto &nn_old = nnhd.nn_old;
            auto it = nnhd.pool.begin();
            for (unsigned l = 0; l < nnhd.M; ++l, it++) {
                auto nn = const_cast<Neighbor *>(&*it);
                auto &nhood_o = graph_[nn->id];  // nn on the other side of the edge

                if (nn->flag) {
                    nn_new.push_back(nn->id);
                    if (nn->distance > nhood_o.pool.rbegin()->distance) {
                        LockGuard guard(nhood_o.lock);
                        nhood_o.cnt_rnew++;
                        if (nhood_o.rnn_new.size() < S)nhood_o.rnn_new.push_back(n);
                        else if (rand() * nhood_o.cnt_rnew < S * RAND_MAX) {
                            unsigned int pos = rand() % S;
                            nhood_o.rnn_new[pos] = n;
                        }
                    }
                    nn->flag = false;
                } else {
                    nn_old.push_back(nn->id);
                    if (nn->distance > nhood_o.pool.rbegin()->distance) {
                        LockGuard guard(nhood_o.lock);
                        nhood_o.cnt_rold++;
                        if (nhood_o.rnn_old.size() < S)nhood_o.rnn_old.push_back(n);
                        else if (rand() * nhood_o.cnt_rold < S * RAND_MAX) {
                            unsigned int pos = rand() % S;
                            nhood_o.rnn_old[pos] = n;
                        }
                    }
                }
            }
        }
#pragma omp parallel for
        for (unsigned i = 0; i < N; ++i) {
            auto &nn_new = graph_[i].nn_new;
            auto &nn_old = graph_[i].nn_old;
            auto &rnn_new = graph_[i].rnn_new;
            auto &rnn_old = graph_[i].rnn_old;
//            if (R && rnn_new.size() > R) {
//                std::random_shuffle(rnn_new.begin(), rnn_new.end());
//                rnn_new.resize(R);
//            }
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
//            if (R && rnn_old.size() > R) {
//                std::random_shuffle(rnn_old.begin(), rnn_old.end());
//                rnn_old.resize(R);
//            }
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
//            if (nn_old.size() > R * 2) {
//                nn_old.resize(R * 2);
////                nn_old.reserve(R * 2);
//            }
            graph_[i].rnn_new.clear();
            graph_[i].rnn_old.clear();
        }
    }

    void IndexGraph::NNDescent(const Parameters &parameters) {
        auto iter = parameters.Get<unsigned>("iter");
        auto K = parameters.Get<unsigned>("K");
        double stop = 0.95;
//        std::mt19937 rng(rand());
//        std::vector<unsigned> control_points(_CONTROL_NUM);
//  std::vector<std::vector<unsigned> > acc_eval_set(_CONTROL_NUM);
//  generate_control_set(control_points, acc_eval_set, N);
//        GenRandom(rng, &control_points[0], _CONTROL_NUM, N);
        vector<unsigned> control_points(N);
        for (unsigned i = 0; i < N; i++)
            control_points[i] = i;
        double recall = eval_recall(control_points, 0.1, K, graph_truth);
        printf("init recall:%.4f\n", recall);
        addRecord("init_recall", dtos(recall, 4));
        unsigned it = 0;
    //    vector<vector<string>> skyline;
//        getSkyline(skyline);
        for (; it < iter; it++) {
            timmer("e_refine");
            addRecord("iter", to_string(it));
            addRecord("total_recall", dtos(recall, 4));
            addRecord("nn_comp", to_string(nn_comp));
            addRecord("nn_time", dtos(timeby("s_refine", "e_refine"), 1));
            addRecord("total_time", dtos(timeby("s_init", "e_refine"), 1));
//            if (record["init_comp"] != "")
//                addRecord("total_comp", to_string(stoll(record["init_comp"]) + stoll(record["nn_comp"])));
//            if (record["db"] == "y" && !dominate(recall, timeby("s_init", "e_refine"), skyline)) {
//                DBexec();
//                printf("new skyline: recall:%.4f total_time:%.1fs\n", recall, timeby("s_init", "e_refine"));
//            } else


            if ( recall > 0.99)
                break;

            timmer("s_descent" + to_string(it));
            update(parameters);
            join();
            timmer("e_descent" + to_string(it));
            recall = eval_recall(control_points, 0.1, K, graph_truth);
            printf("iter:%d recall:%.4f\ttime:%.1fs\ttotal_time:%.1fs\n", it, recall,
                   timeby("s_descent" + to_string(it), "e_descent" + to_string(it)),
                   timeby("s_init", "e_descent" + to_string(it)));
            addRecord("iter" + to_string(it) + "_recall", dtos(recall, 4));
            addRecord("iter" + to_string(it) + "_time",
                      dtos(timeby("s_descent" + to_string(it), "e_descent" + to_string(it)), 1));
        }
        recall = eval_recall(control_points, 1, K, graph_truth);
        printf("total_recall:%.4f\n", recall);
        printf("nn_comp:%llu\n", nn_comp);
    }

    void IndexGraph::generate_control_set(std::vector<unsigned> &c,
                                          std::vector<std::vector<unsigned> > &v,
                                          unsigned N) {
#pragma omp parallel for
        for (unsigned i = 0; i < c.size(); i++) {
            std::vector<Neighbor> tmp;
            for (unsigned j = 0; j < N; j++) {
                float dist = distance_->compare(data_ + c[i] * dim_, data_ + j * dim_, dim_);
                tmp.emplace_back(j, dist, true);
            }
            std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
            for (unsigned j = 0; j < _CONTROL_NUM; j++) {
                v[i].push_back(tmp[j].id);
            }
        }
    }

    double
    IndexGraph::eval_recall(std::vector<unsigned> &ctrl_points, double p, unsigned K, const unsigned *acc_eval_set) {
        double acc = 0;
        unsigned cnt = 0;
        for (unsigned i : ctrl_points) {
            if (rand() > p * RAND_MAX)
                continue;
            cnt++;
            auto &pool = graph_[i].pool;
            for (unsigned k = 0; k < K; k++) {
                auto &val = acc_eval_set[i * truthNum + k];
                for (auto &j : pool) {
                    if (j.id == val) {
                        acc++;
                        break;
                    }
                }
            }
        }
        return acc / (cnt * K);
    }


//    void IndexGraph::InitializeGraph(const Parameters &parameters) {
//
//        const unsigned L = parameters.Get<unsigned>("L");
//        const unsigned S = parameters.Get<unsigned>("S");
//
//        graph_.reserve(N);
//        std::mt19937 rng(rand());
//        for (unsigned i = 0; i < N; i++) {
//            graph_.push_back(nhood(L, S, rng, (unsigned) N));
//        }
//#pragma omp parallel for
//        for (unsigned i = 0; i < N; i++) {
//            const float *query = data_ + i * dim_;
//            std::vector<unsigned> tmp(S + 1);
//            initializer_->Search(query, data_, S + 1, parameters, tmp.data());
//
//            for (unsigned j = 0; j < S; j++) {
//                unsigned id = tmp[j];
//                if (id == i)continue;
//                float dist = distance_->compare(data_ + i * dim_, data_ + id * dim_, (unsigned) dim_);
//                graph_[i].pool.push_back(Neighbor(id, dist, true));
//            }
//            std::make_heap(graph_[i].pool.begin(), graph_[i].pool.end());
//            graph_[i].pool.reserve(L);
//        }
//    }

    void IndexGraph::InitializeGraph_Refine(const Parameters &parameters) {
        assert(final_graph_.size() == N);

        const unsigned L = parameters.Get<unsigned>("L");
        const unsigned S = parameters.Get<unsigned>("S");

        graph_.reserve(N);
        std::mt19937 rng(rand());
        for (unsigned i = 0; i < N; i++) {
            graph_.push_back(nhood(L, S, rng, (unsigned) N));
        }
#pragma omp parallel for
        for (unsigned i = 0; i < N; i++) {
            auto &ids = final_graph_[i];
            std::sort(ids.begin(), ids.end());

            size_t K = ids.size();

            for (unsigned j = 0; j < K; j++) {
                unsigned id = ids[j];
                if (id == i || (j > 0 && id == ids[j - 1]))continue;
                float dist = distance_->compare(data_ + i * dim_, data_ + id * dim_, (unsigned) dim_);
                graph_[i].pool.insert(Neighbor(id, dist, true));
            }
            std::vector<unsigned>().swap(ids);
        }
        CompactGraph().swap(final_graph_);
    }


    void IndexGraph::RefineGraph(const float *data, const Parameters &parameters) {
        data_ = data;
        assert(initializer_->HasBuilt());

        InitializeGraph_Refine(parameters);
        NNDescent(parameters);

//        final_graph_.reserve(N);
//        unsigned K = parameters.Get<unsigned>("K");
//        for (unsigned i = 0; i < N; i++) {
//            std::vector<unsigned> tmp;
//            auto it = graph_[i].pool.begin();
//            for (unsigned j = 0; j < K; j++, it++) {
//                tmp.push_back(it->id);
//            }
//            tmp.reserve(K);
//            final_graph_.push_back(tmp);
//            std::set<Neighbor>().swap(graph_[i].pool);
//            std::vector<unsigned>().swap(graph_[i].nn_new);
//            std::vector<unsigned>().swap(graph_[i].nn_old);
//            std::vector<unsigned>().swap(graph_[i].rnn_new);
//            std::vector<unsigned>().swap(graph_[i].rnn_new);
//        }
//        std::vector<nhood>().swap(graph_);
        has_built = true;

    }


    void IndexGraph::Build(size_t n, const float *data, const Parameters &parameters) {
//
//        //assert(initializer_->GetDataset() == data);
//        data_ = data;
//        assert(initializer_->HasBuilt());
//
//        InitializeGraph(parameters);
//        NNDescent(parameters);
//        //RefineGraph(parameters);
//
//        final_graph_.reserve(N);
//        std::cout << N << std::endl;
//        unsigned K = parameters.Get<unsigned>("K");
//        for (unsigned i = 0; i < N; i++) {
//            std::vector<unsigned> tmp;
//            std::sort(graph_[i].pool.begin(), graph_[i].pool.end());
//            for (unsigned j = 0; j < K; j++) {
//                tmp.push_back(graph_[i].pool[j].id);
//            }
//            tmp.reserve(K);
//            final_graph_.push_back(tmp);
//            std::vector<Neighbor>().swap(graph_[i].pool);
//            std::vector<unsigned>().swap(graph_[i].nn_new);
//            std::vector<unsigned>().swap(graph_[i].nn_old);
//            std::vector<unsigned>().swap(graph_[i].rnn_new);
//            std::vector<unsigned>().swap(graph_[i].rnn_new);
//        }
//        std::vector<nhood>().swap(graph_);
//        has_built = true;
    }

    void IndexGraph::Search(
            const float *query,
            const float *x,
            size_t K,
            const Parameters &parameter,
            unsigned *indices) {
        const unsigned L = parameter.Get<unsigned>("L_search");

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned) N);

        std::vector<char> flags(N);
        memset(flags.data(), 0, N * sizeof(char));
        for (unsigned i = 0; i < L; i++) {
            unsigned id = init_ids[i];
            float dist = distance_->compare(data_ + dim_ * id, query, (unsigned) dim_);
            retset[i] = Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id])continue;
                    flags[id] = 1;
                    float dist = distance_->compare(query, data_ + dim_ * id, (unsigned) dim_);
                    if (dist >= retset[L - 1].distance)continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    //if(L+1 < retset.size()) ++L;
                    if (r < nk)nk = r;
                }
                //lock to here
            }
            if (nk <= k)k = nk;
            else ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexGraph::Save(const char *filename) {
        std::ofstream out(filename, std::ios::binary | std::ios::out);
        assert(final_graph_.size() == N);
        unsigned GK = (unsigned) final_graph_[0].size();
        for (unsigned i = 0; i < N; i++) {
            out.write((char *) &GK, sizeof(unsigned));
            out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
        }
        out.close();
    }

    void IndexGraph::Load(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        unsigned k;
        in.read((char *) &k, 4);
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t) ss;
        size_t num = fsize / ((size_t) k + 1) / 4;
        in.seekg(0, std::ios::beg);

        final_graph_.resize(num);
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);
            final_graph_[i].resize(k);
            final_graph_[i].reserve(k);
            in.read((char *) final_graph_[i].data(), k * sizeof(unsigned));
        }
        in.close();
    }

    void IndexGraph::parallel_graph_insert(unsigned id, Neighbor nn, LockGraph &g, size_t K) {
        LockGuard guard(g[id].lock);
        size_t l = g[id].pool.size();
        if (l == 0)g[id].pool.insert(nn);
        else {
//            InsertIntoPool(g[id].pool.data(), (unsigned) l, nn);
//            if (g[id].pool.size() > K)g[id].pool.reserve(K);
        }

    }

    void IndexGraph::GraphAdd(const float *data, unsigned n_new, unsigned dim, const Parameters &parameters) {
        data_ = data;
        data += N * dim_;
        assert(final_graph_.size() == N);
        assert(dim == dim_);
        unsigned total = n_new + (unsigned) N;
        LockGraph graph_tmp(total);
        size_t K = final_graph_[0].size();
        compact_to_Lockgraph(graph_tmp);
        unsigned seed = 19930808;
#pragma omp parallel
        {
            std::mt19937 rng(seed ^ omp_get_thread_num());
#pragma omp for
            for (unsigned i = 0; i < n_new; i++) {
                std::vector<Neighbor> res;
                get_neighbor_to_add(data + i * dim, parameters, graph_tmp, rng, res, n_new);

                for (unsigned j = 0; j < K; j++) {
                    parallel_graph_insert(i + (unsigned) N, res[j], graph_tmp, K);
                    parallel_graph_insert(res[j].id, Neighbor(i + (unsigned) N, res[j].distance, true), graph_tmp, K);
                }

            }
        };


        std::cout << "complete: " << std::endl;
        N = total;
        final_graph_.resize(total);
        for (unsigned i = 0; i < total; i++) {
            auto it = graph_tmp[i].pool.begin();
            for (unsigned m = 0; m < K; m++, it++) {
                final_graph_[i].push_back(it->id);
            }
        }

    }

    void IndexGraph::get_neighbor_to_add(const float *point,
                                         const Parameters &parameters,
                                         LockGraph &g,
                                         std::mt19937 &rng,
                                         std::vector<Neighbor> &retset,
                                         unsigned n_new) {
        const unsigned L = parameters.Get<unsigned>("L_ADD");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        GenRandom(rng, init_ids.data(), L / 2, n_new);
        for (unsigned i = 0; i < L / 2; i++)init_ids[i] += N;

        GenRandom(rng, init_ids.data() + L / 2, L - L / 2, (unsigned) N);

        unsigned n_total = (unsigned) N + n_new;
        std::vector<char> flags(n_new + n_total);
        memset(flags.data(), 0, n_total * sizeof(char));
        for (unsigned i = 0; i < L; i++) {
            unsigned id = init_ids[i];
            float dist = distance_->compare(data_ + dim_ * id, point, (unsigned) dim_);
            retset[i] = Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                LockGuard guard(g[n].lock);//lock start
                auto it = g[n].pool.begin();
                for (unsigned m = 0; m < g[n].pool.size(); ++m, ++it) {
                    unsigned id = it->id;
                    if (flags[id])continue;
                    flags[id] = 1;
                    float dist = distance_->compare(point, data_ + dim_ * id, (unsigned) dim_);
                    if (dist >= retset[L - 1].distance)continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    //if(L+1 < retset.size()) ++L;
                    if (r < nk)nk = r;
                }
                //lock to here
            }
            if (nk <= k)k = nk;
            else ++k;
        }


    }

    void IndexGraph::compact_to_Lockgraph(LockGraph &g) {

        //g.resize(final_graph_.size());
        for (unsigned i = 0; i < final_graph_.size(); i++) {
            for (unsigned j = 0; j < final_graph_[i].size(); j++) {
                float dist = distance_->compare(data_ + i * dim_,
                                                data_ + final_graph_[i][j] * dim_, (unsigned) dim_);
                g[i].pool.insert(Neighbor(final_graph_[i][j], dist, true));
            }
            std::vector<unsigned>().swap(final_graph_[i]);
        }
        CompactGraph().swap(final_graph_);
    }


}
