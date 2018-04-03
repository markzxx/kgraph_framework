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
        tablelen_ = params_.Get<unsigned>("codelen");
        numTable_ = params_.Get<unsigned>("numTable");
        codelen_ = numTable_ * tablelen_;

    }

    IndexLSH::~IndexLSH() {}

    void IndexLSH::Build() {

        projection_matrix = new float[codelen_ * dim_];
#ifdef linux
        ProfilerStart("my.prof");
#endif
        timmer("s_build");
        generate_random_projection_matrix(dim_, codelen_, projection_matrix);
        random_projection(data_, N, dim_, codelen_, BaseCode);
        timmer("e_projection");
        output_time("Projection time", "s_build", "e_projection");
        navie_bucketIndex();
//        timmer("e_bucket");
//        output_time("Bucket time","e_projection","e_bucket");
#ifdef linux
        ProfilerStop();
#endif
        has_built = true;
    }

    void IndexLSH::Build(size_t n, const float *data, const Parameters &parameters) {}

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


    void IndexLSH::navie_bucketIndex2() {
        unsigned L = params_.Get<unsigned>("L");
        unsigned K = params_.Get<unsigned>("K");

        buildHashTable();

        timmer("s_bucket");
        vector<unordered_set<unsigned >> navieBucket(N);
        unordered_set<unsigned> keyset;
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            keyset.insert(BaseCode[0][i]);
            if ((i + 1) % (N / 10) == 0)
                cout << i * 10 / N << " ";
            Codes &baseCode = BaseCode;
            for (int j = 0; j < numTable_; j++) {
                Bucket &bucket = hashtable[j];
                auto r = bucket.equal_range(baseCode[j][i]);
                for (auto it = r.first; it != r.second; it++) {
                    if (it->second != i)
                        navieBucket[i].insert(it->second);
//                    if (navieBucket[i].size() > K)
//                        break;
                }
//                if (navieBucket[i].size() > K)
//                    break;
            }
        }
        cout << endl << N / keyset.size() << endl;

        clearHashtable();

        final_graph_.resize(N);
        std::mt19937 rng((unsigned) time(nullptr));
        size_t dif = 0;
#pragma omp parallel for
        for (unsigned i = 0; i < N; i++) {
            std::vector<unsigned> tmp;
            auto &bucket = navieBucket[i];
            bucket.erase(i);
            dif += bucket.size();
            while (bucket.size() < K)
                bucket.insert(rng() % N);
            auto it = bucket.begin();
            while (tmp.size() < K)
                tmp.push_back(*it++);
            final_graph_[i] = tmp;
            bucket.clear();
        }
        timmer("e_bucket");
        output_time("bucket time", "s_bucket", "e_bucket");
        printf("avg bucket num:%d\n", dif / N);
    }

    void IndexLSH::vec_navie_bucketIndex() {
        unsigned L = params_.Get<unsigned>("L");
        unsigned K = params_.Get<unsigned>("K");

        vector<unordered_map<unsigned, vector<unsigned >>> hashtable;
        hashtable.reserve(numTable_);
        for (unsigned j = 0; j < numTable_; j++) {
            unordered_map<unsigned, vector<unsigned >> table;
            for (unsigned i = 0; i < N; i++) {
                auto it = table.insert({BaseCode[j][i], vector<unsigned>()});
                it.first->second.push_back(i);
            }
            hashtable.push_back(table);
        }

        timmer("s_bucket");
        vector<unordered_map<unsigned, unsigned >> navieBucket(N);
        int numbuc = 0;
#pragma omp parallel for
        for (int t = 0; t < numTable_; t++) {
            cout << t << " ";
            if ((t + 1) % 8 == 0)
                cout << endl;
            unordered_map<unsigned, vector<unsigned >> &bucket = hashtable[t];
            for (auto &it : bucket) {
                auto &vec = it.second;
                auto num = vec.size();
                numbuc += num;
                for (auto i = 0; i < num; i++)
                    for (auto j = i + 1; j < num; j++) {
                        auto mapi = navieBucket[i].insert({j, 1});
                        if (!mapi.second)
                            mapi.first->second += 1;

                        auto mapj = navieBucket[j].insert({i, 1});
                        if (!mapj.second)
                            mapj.first->second += 1;
                    }
            }
        }
        cout << (N / hashtable[0].size()) << endl;
        clearHashtable();
        size_t dif = 0;
        final_graph_.resize(N);
        std::mt19937 rng(0);
#pragma omp parallel for
        for (unsigned i = 0; i < N; i++) {
            std::vector<pair<unsigned, unsigned >> tmp;
            tmp.reserve(K);
            auto &bucket = navieBucket[i];
            bucket.erase(i);
            dif += bucket.size();
            while (bucket.size() < K)
                bucket.insert({rng() % N, 1});
            auto it = bucket.begin();
            for (int k = 0; k < K; ++k, ++it) {
                tmp.emplace_back(make_pair(it->second, it->first));
            }
            if (bucket.size() > K)
                make_heap(tmp.begin(), tmp.end());
            for (; it != bucket.end(); ++it) {
                if (it->second > tmp.front().first) {
                    pop_heap(tmp.begin(), tmp.end());
                    tmp.back() = make_pair(it->second, it->first);
                    push_heap(tmp.begin(), tmp.end());
                }
            }

            final_graph_[i].reserve(K);
            for (int f = 0; f < K; f++)
                final_graph_[i].push_back(tmp[f].second);
            bucket.clear();
        }
        timmer("e_bucket");
        output_time("bucket time", "s_bucket", "e_bucket");
        printf("avg bucket num:%d\n", dif / N);
    }

    void IndexLSH::navie_bucketIndex() {
        unsigned L = params_.Get<unsigned>("L");
        unsigned K = params_.Get<unsigned>("K");

        buildHashTable();

        timmer("s_bucket");
        vector<unordered_map<unsigned, unsigned >> navieBucket(N);
        for (auto &b : navieBucket)
            b.reserve(K);
        unordered_set<unsigned> keyset;
//#pragma omp parallel for
        for (int i = 0; i < numTable_; i++) {
            cout << i << " ";
            if ((i + 1) % 8 == 0)
                cout << endl;
            Bucket &bucket = hashtable[i];
            for (auto iti = bucket.begin(); iti != bucket.end();) {
                auto key = iti->first;
                keyset.insert(key);
                unsigned vi = iti->second;
                for (auto itj = ++iti; itj != bucket.end() && itj->first == key; itj++) {
                    unsigned vj = iti->second;
                    //vector
//                    if(navieBucket[vi].size()<K)
//                        navieBucket[vi].push_back(vj);
////                    else{
////                        unsigned int pos = rand() % K;
////                        navieBucket[vi][pos] = vj;
////                    }
//                    if(navieBucket[vj].size()<K)
//                        navieBucket[vj].push_back(vi);
////                    else{
////                        unsigned int pos = rand() % K;
////                        navieBucket[vj][pos] = vi;
////                    }

                    //hashmap
                    auto mapi = navieBucket[vi].insert({vj, 1});
                    if (!mapi.second)
                        mapi.first->second += 1;

                    auto mapj = navieBucket[vj].insert({vi, 1});
                    if (!mapj.second)
                        mapj.first->second += 1;
                }
            }
        }
        cout << endl << N / keyset.size() << endl;
        clearHashtable();
        size_t dif = 0;
        final_graph_.resize(N);
        std::mt19937 rng(0);
#pragma omp parallel for
        for (unsigned i = 0; i < N; i++) {
            dif += navieBucket[i].size();

            //vector
//            while (navieBucket[i].size()<K)
//                navieBucket[i].push_back(rng()%N);
//            navieBucket[i].resize(K);
//            final_graph_.push_back(navieBucket[i]);

            vector<pair<unsigned, unsigned >> tmp;
            tmp.reserve(K);
            auto &bucket = navieBucket[i];
//            map<unsigned, unsigned > tmp_map;
//            for(auto b : bucket){
//                auto mapi = tmp_map.insert({b,1});
//                if(!mapi.second)
//                    mapi.first->second+=1;
//            }

//            while (tmp_map.size() < K)
//                tmp_map.insert({rng() % N,1});
//            auto it = tmp_map.begin();
            while (bucket.size() < K)
                bucket.insert({rng() % N, 1});
            auto it = bucket.begin();
            for (int k = 0; k < K; ++k, ++it) {
                tmp.emplace_back(make_pair(it->second, it->first));
            }
            if (bucket.size() > K)
                make_heap(tmp.begin(), tmp.end());
            for (; it != bucket.end(); ++it) {
                if (it->second > tmp.front().first) {
                    pop_heap(tmp.begin(), tmp.end());
                    cout << tmp.back().first;
//                    exit(-1);
                    tmp.back() = make_pair(it->second, it->first);
                    cout << " " << tmp.back().first << endl;
                    push_heap(tmp.begin(), tmp.end());
                }
            }
            final_graph_[i].reserve(K);
            for (int f = 0; f < K; f++)
                final_graph_[i].push_back(tmp[f].second);
            bucket.clear();
        }
        timmer("e_bucket");
        output_time("bucket time", "s_bucket", "e_bucket");
        printf("avg bucket num:%d\n", dif / N);
//        exit(-1);
    }

    void IndexLSH::bucketIndex() {
        unsigned L = params_.Get<unsigned>("L");
        unsigned K = params_.Get<unsigned>("K");

        buildHashTable();

        timmer("s_bucket");
        knn_graph.resize(N);
#pragma omp parallel for
        for (int i = 0; i < numTable_; i++) {
            cout << i << " ";
            if (i % 8 == 0)
                cout << endl;
            Codes &baseCode = BaseCode;
            CompactGraph &final = final_graph_;
            const float *data = data_;
            Bucket &bucket = hashtable[i];
            for (auto iti = bucket.begin(); iti != bucket.end();) {
                auto key = iti->first;
//                cout << key <<endl;
                unsigned vi = iti->second;
                for (auto itj = ++iti; itj != bucket.end() && itj->first == key; itj++) {
                    unsigned vj = iti->second;
                    float dist = -1;
                    Candidate1 cj(vj, dist);
                    if (knn_graph[vi].find(cj) == knn_graph[vi].end()) {
                        dist = distance_->compare(data + vi * dim_, data_ + vj * dim_, dim_);
                        cj.distance = dist;
                        if (knn_graph[vi].size() < K) {
                            knn_graph[vi].insert(cj);
                        } else if (dist < knn_graph[vi].begin()->distance) {
                            knn_graph[vi].erase(knn_graph[vi].begin());
                            knn_graph[vi].insert(cj);
                        }
                    }

//                    knn_graph[vi].insert(vj, dist);
                    Candidate1 ci(vi, dist);
                    if (knn_graph[vj].find(ci) == knn_graph[vj].end()) {
                        if (dist < 0) {
                            dist = distance_->compare(data + vi * dim_, data_ + vj * dim_, dim_);
                            ci.distance = dist;
                        }
                        if (knn_graph[vj].size() < K) {
                            knn_graph[vj].insert(ci);
                        } else if (dist < knn_graph[vj].begin()->distance) {
                            knn_graph[vj].erase(knn_graph[vj].begin());
                            knn_graph[vj].insert(ci);
                        }
                    }
//                    knn_graph[vj].insert(vi, dist);
                }
            }
        }
        clearHashtable();
        size_t dif = 0;
        final_graph_.resize(N);
        std::mt19937 rng((unsigned) time(nullptr));
        std::set<unsigned> result;
#pragma omp parallel for
        for (unsigned i = 0; i < N; i++) {
            auto &bucket = knn_graph[i];
            while (bucket.size() < K)
                bucket.insert(Candidate1(rng() % N, 0));
            std::vector<unsigned> tmp;
            for (auto it = knn_graph[i].begin(); it != knn_graph[i].end(); it++) {
                tmp.push_back(it->id);
            }
            final_graph_[i] = tmp;
            bucket.clear();
        }

        timmer("e_bucket");
        output_time("bucket time", "s_bucket", "e_bucket");
        printf("avg dif:%d\n", dif / N);
    }

    void IndexLSH::init_graph() {
        unsigned L = params_.Get<unsigned>("L");
        unsigned K = params_.Get<unsigned>("K");
        final_graph_.resize(N);
#pragma omp parallel for
        for (size_t cur = 0; cur < N; cur++) {
//            cout<<cur<<endl;
            std::vector<int> hammingDistance(N);

            for (size_t j = 0; j < numTable_; j++) {
                for (size_t i = 0; i < N; i++) {
                    hammingDistance[i] += parallel_popcnt32(BaseCode[j][i] ^ BaseCode[j][cur]);
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
            final_graph_[cur] = res;
        }
        cout << "Init graph finished" << endl;
    }

    void IndexLSH::generate_random_projection_matrix(int dim, int codelen, float *projection_matrix) {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-1.0, 1.0);
#pragma omp parallel for
        for (int i = 0; i < codelen * dim; i++) {
            projection_matrix[i] = distribution(generator);
        }
    }

    void
    IndexLSH::random_projection(const float *data, size_t points_num, unsigned dim, unsigned codelen, Codes &output) {
        float *projection_result = new float[codelen * points_num];

        cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, points_num, codelen, dim,
                    1, data, dim, projection_matrix, codelen, 0, projection_result, points_num);
//        DistanceInnerProduct dist;
//        for(int i=0;i<N;i++){
//            projection_result[i]=dist.compare(data+i*dim,projection_matrix,dim);
//        }
        output.resize(numTable_);

#pragma omp parallel for
        for (unsigned i = 0; i < numTable_; i++) {
            output[i].resize(points_num);
        }

#pragma omp parallel for
        for (unsigned i = 0; i < codelen; i++) {
            const int table_id = i / tablelen_;
            const unsigned bit = 1u << (i % tablelen_);
            for (size_t j = 0; j < points_num; j++) {
                if (projection_result[i * points_num + j] > 0)
                    output[table_id][j] |= bit;
            }
        }
        cout << "Hashing finished" << endl;
        delete[] projection_result;
        delete[] projection_matrix;
    }

    void IndexLSH::buildHashTable() {
        timmer("s_hashtable");

#pragma omp parallel for
        for (unsigned j = 0; j < numTable_; j++) {
            Bucket table;
            for (unsigned i = 0; i < N; i++) {
                table.insert({BaseCode[j][i], i});
            }
            hashtable.push_back(table);
        }

        timmer("e_hashtable");
        output_time("Hashtable time", "s_hashtable", "e_hashtable");
    }

}