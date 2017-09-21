#include <vector>
#include <functional>
#include <random>
#include <chrono>
#include <algorithm>
#include <cassert>

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

template<typename T>
ssize_t linear_search(std::vector<T> const &values, T key)
{
    size_t n = values.size();
    const T *d = values.data();
    for (size_t i = 0; i < n; i++) {
        if (d[i] == key)
            return i;
    }
    return -1;
}

template<typename T>
ssize_t naive_search(std::vector<T> const &values, T key)
{
    ssize_t low = 0, hi = values.size();
    const T *d = values.data();
    while (low < hi) {
        ssize_t mid = low + (hi-low)/2;
        int v = d[mid];
        if (key == v)
            return mid;
        else if (key < v)
            hi = mid;
        else
            low = mid + 1;
    }
    return -1;
}

// http://databasearchitects.blogspot.jp/2015/09/trying-to-speed-up-binary-search.html
template<typename T>
ssize_t short_search(std::vector<T> const &values, T needle)
{
    size_t n = values.size();
    const T *lower = values.data();
    while (size_t half=n/2) {
        const T *middle=lower+half;
        lower=((*middle)<=needle)?middle:lower;
        n-=half;
    }
    return ((*lower)==needle)?lower-values.data():-1;
}

ssize_t dump_short_search(std::vector<int> const &values, int key)
{
    return short_search(values, key);
}

template<typename T>
ssize_t std_search(std::vector<T> const &values, T key)
{
    typename std::vector<T>::const_iterator end = values.end();
    typename std::vector<T>::const_iterator it = std::lower_bound(values.begin(), end, key);
    //return (it != end && *it == key) ? std::distance(values.begin(), it) : -1;
    return std::distance(values.begin(), it);
}

// assuming values.size() is power of two
// See comments under https://schani.wordpress.com/2010/04/30/linear-vs-binary-search/
template<typename T>
ssize_t pot_search(std::vector<T> const &values, T key)
{
    ssize_t i = 0, step;
    for (step = values.size() / 2; step > 0; step >>= 1)
        if (values[i | step] <= key)
            i |= step;
    return i;
}

template <class T>
inline size_t choose(T a, T b, size_t src1, size_t src2)
{
#if defined(__clang__) && defined(__x86_64) && 0 // my clang cannot compile this asm code
    size_t res = src1;
    asm("cmpq %1, %2; cmovaeq %4, %0"
        :
    "=q" (res)
        :
        "q" (a),
        "q" (b),
        "q" (src1),
        "q" (src2),
        "0" (res)
        :
        "cc");
    return res;
#else
    return b >= a ? src2 : src1;
#endif
}

// https://realm.io/news/how-we-beat-cpp-stl-binary-search/
// Unroll version 3, fast_upper_bound4
template <class T>
inline size_t unroll3_search(const std::vector<T>& vec, T value)
{
    size_t size = vec.size();
    size_t low = 0;

    while (size >= 8) {
        size_t half = size / 2;
        size_t other_half = size - half;
        size_t probe = low + half;
        size_t other_low = low + other_half;
        T v = vec[probe];
        size = half;
        low = choose(v, value, low, other_low);

        half = size / 2;
        other_half = size - half;
        probe = low + half;
        other_low = low + other_half;
        v = vec[probe];
        size = half;
        low = choose(v, value, low, other_low);

        half = size / 2;
        other_half = size - half;
        probe = low + half;
        other_low = low + other_half;
        v = vec[probe];
        size = half;
        low = choose(v, value, low, other_low);
    }

    while (size > 0) {
        size_t half = size / 2;
        size_t other_half = size - half;
        size_t probe = low + half;
        size_t other_low = low + other_half;
        T v = vec[probe];
        size = half;
        low = choose(v, value, low, other_low);
    };

    return low;
}

ssize_t dump_unroll3_search(std::vector<int> const &values, int key)
{
    return unroll3_search(values, key);
}

template<typename Search>
void benchmark(const std::vector<int> &keys, const std::vector<int> &values, Search search, const char *desc)
{
    ssize_t sum = 0;
    const auto tbegin = std::chrono::high_resolution_clock::now();
    for (auto &k : keys) {
        ssize_t i = search(values, k);
        assert(values[i] == k);
        sum += i; // avoid optimization
    }
    const auto tend = std::chrono::high_resolution_clock::now();
    const auto timespan = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count();
    printf("[%s] sum: %zd, time: %" PRId64 "\n", desc, sum, timespan);
}

static void benchmarkSuite(int numKeys, int numValues)
{
    printf("keys: %d, values: %d\n", numKeys, numValues);
    std::mt19937 rng(303);
    std::uniform_int_distribution<int> keyDist(0, numValues-1);
    std::uniform_int_distribution<int> valueDist(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    std::vector<int> keys(numKeys);
    std::vector<int> values(numValues * 1.2); // in case of duplication
    for (auto &v : values)
        v = valueDist(rng);
    std::sort(values.begin(), values.end());

    values.erase(std::unique(values.begin(), values.end()), values.end());
    printf("values: %zu\n", values.size());
    assert((int)values.size() >= numValues);
    if (values.size() > (size_t)numValues) { // remove extra
        values.erase(values.begin() + numValues, values.end());
    }
    printf("values: %zu\n", values.size());

    for (auto &k : keys)
        k = values[keyDist(rng)];

    benchmark(keys, values, naive_search<int>, "naive_search");
    benchmark(keys, values, short_search<int>, "short_search");
    benchmark(keys, values, pot_search<int>, "pot_search");
    benchmark(keys, values, unroll3_search<int>, "unroll3_search");
    benchmark(keys, values, std_search<int>, "std_search");
    if (numValues < 10000) {
        benchmark(keys, values, linear_search<int>, "linear_search");
    }
}

int main()
{
    int kNumKeys = 1000000;
    benchmarkSuite(kNumKeys, 1<<5);
    benchmarkSuite(kNumKeys, 1<<6);
    benchmarkSuite(kNumKeys, 1<<8);
    benchmarkSuite(kNumKeys, 1<<10);
    benchmarkSuite(kNumKeys, 1<<11);
    benchmarkSuite(kNumKeys, 1<<12);
    benchmarkSuite(kNumKeys, 1<<13);
    benchmarkSuite(kNumKeys, 1<<17);
    benchmarkSuite(kNumKeys, 1<<18);
    benchmarkSuite(kNumKeys, 1<<19);
    return 0;
}
