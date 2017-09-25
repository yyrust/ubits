#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <smmintrin.h>
#include <immintrin.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

typedef uint32_t DocId;
typedef std::vector<DocId> PostingList;

const DocId kInvalidDocId = 0xFFFFFFFFu;

static void generateList(DocId maxId, DocId length, PostingList &list)
{
    assert(length <= maxId + 1);
    std::mt19937 rng(maxId ^ length);
    std::uniform_int_distribution<DocId> idDist(0, maxId);
    for (DocId i = 0; i < length; i++)
        list.push_back(idDist(rng));
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
}

static void linearIntersection(PostingList const &L1, PostingList const &L2, PostingList &out)
{
    PostingList::const_iterator i1 = L1.begin(), end1 = L1.end();
    PostingList::const_iterator i2 = L2.begin(), end2 = L2.end();
    while (i1 != end1 && i2 != end2) {
        if (*i1 == *i2) {
            out.push_back(*i1);
            ++i1;
            ++i2;
        }
        else if (*i1 < *i2) {
            ++i1;
        }
        else {
            ++i2;
        }
    }
}

struct BinarySeek
{
    static DocId seek(PostingList const &list, size_t &start, size_t end, const DocId id)
    {
        size_t lo = start, hi = end;
        while (lo < hi) {
            const size_t mid = (hi - lo)/2 + lo;
            const DocId v = list[mid];
            if (id == v) {
                start = mid;
                return id;
            }
            else if (id < v) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        start = lo;
        if (lo == list.size()) {
            return kInvalidDocId;
        }
        else {
            return list[lo];
        }
    }

    static DocId seek(PostingList const &list, size_t &start, const DocId id)
    {
        size_t lo = start, hi = list.size();
        while (lo < hi) {
            const size_t mid = (hi - lo)/2 + lo;
            const DocId v = list[mid];
            if (id == v) {
                start = mid;
                return id;
            }
            else if (id < v) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        start = lo;
        if (lo == list.size()) {
            return kInvalidDocId;
        }
        else {
            return list[lo];
        }
    }
};

struct SimpleBinarySeek
{
    static DocId seek(PostingList const &list, size_t &start, size_t end, const DocId id)
    {
        const DocId *first = list.data() + start;
        const DocId *low = first;
        size_t n = end - start;
        size_t half;
        while ((half = n / 2) > 0) {
            const DocId *mid = low + half;
            low = (*mid < id) ? mid : low;
            n -= half;
        }
        start = low - list.data();
        if (start == list.size()) {
            return kInvalidDocId;
        }
        else {
            ++start;
            return *(low+1);
        }
    }
};

template<typename FallbackSeek>
struct GallopingSeek
{
    static DocId seek(PostingList const &list, size_t &start, const DocId id)
    {
        size_t index = start;
        size_t size = list.size();
        assert(index < size);
        if (list.back() < id) {
            start = size;
            return kInvalidDocId;
        }
        if (list[index] >= id) {
            start = index;
            return list[index];
        }
        size_t prev = start;
        size_t step = 1;
        while (index < size) {
            DocId value = list[index];
            if (value == id) {
                start = index;
                return id;
            }
            if (value > id) {
                break;
            }
            prev = index;
            index += step;
            step *= 2;
        }
        // invariant: list[prev] < id
        // invariant: list[index] > id
        if (index >= size) {
            index = size - 1;
        }
        start = prev;
        return FallbackSeek::seek(list, start, index + 1, id);
    }
};

struct AVX2
{
    typedef __m256i value_type;
    typedef uint32_t mask_type;
    const static size_t kVectorSize = 8;

    template<typename T>
    static value_type load(T *array)
    {
        return _mm256_loadu_si256((const value_type *)array);
    }

    static value_type init(DocId id)
    {
        return _mm256_set1_epi32(id);
    }

    static mask_type eq(const value_type &lhs, const value_type &rhs)
    {
        value_type result = _mm256_cmpeq_epi32(lhs, rhs);
        return _mm256_movemask_epi8(result);
    }

    static mask_type lt(const value_type &lhs, const value_type &rhs)
    {
        value_type result = _mm256_cmpgt_epi32(rhs, lhs);
        return _mm256_movemask_epi8(result);
    }

    static mask_type gt(const value_type &lhs, const value_type &rhs)
    {
        value_type result = _mm256_cmpgt_epi32(lhs, rhs);
        return _mm256_movemask_epi8(result);
    }
};

struct SSE4
{
    typedef __m128i value_type;
    typedef uint16_t mask_type;
    const static size_t kVectorSize = 4;

    template<typename T>
    static value_type load(T *array)
    {
        return _mm_loadu_si128((const value_type *)array);
    }

    static value_type init(DocId id)
    {
        return _mm_set1_epi32(id);
    }

    static mask_type eq(const value_type &lhs, const value_type &rhs)
    {
        value_type result = _mm_cmpeq_epi32(lhs, rhs);
        return _mm_movemask_epi8(result);
    }

    static mask_type lt(const value_type &lhs, const value_type &rhs)
    {
        value_type result = _mm_cmplt_epi32(lhs, rhs);
        return _mm_movemask_epi8(result);
    }

    static mask_type gt(const value_type &lhs, const value_type &rhs)
    {
        value_type result = _mm_cmpgt_epi32(lhs, rhs);
        return _mm_movemask_epi8(result);
    }
};

template<typename SIMDTraits>
struct SimdGallopingSeeker
{
    PostingList const &m_list;
    size_t m_alignedSize;
    size_t m_index;
    size_t m_index2;

    SimdGallopingSeeker(PostingList const &list)
    : m_list(list)
    , m_alignedSize(list.size() & ~(SIMDTraits::kVectorSize-1))
    , m_index(0)
    , m_index2(0)
    {
    }

    DocId next()
    {
        m_index2++;
        m_index = m_index2 & ~(SIMDTraits::kVectorSize-1);
        if (m_index2 >= m_list.size())
            return kInvalidDocId;
        else
            return m_list[m_index2];
    }

    DocId seek(const DocId id)
    {
        size_t step = SIMDTraits::kVectorSize;
        size_t index = m_index;
        size_t tail, prev = index;
        while ((tail = index + (SIMDTraits::kVectorSize-1)) < m_alignedSize) {
            if (m_list[tail] >= id) { // found
                // binary search for the first block whose tail >= id
                const DocId *low = m_list.data() + prev;
                size_t n = (index + SIMDTraits::kVectorSize - prev) / SIMDTraits::kVectorSize;
                size_t half;
                while ((half = n / 2) > 0) {
                    const DocId *mid = low + half * SIMDTraits::kVectorSize;
                    low = (*(mid + SIMDTraits::kVectorSize - 1) < id) ? mid : low;
                    n -= half;
                }
                if (prev != index)
                    low += SIMDTraits::kVectorSize;
                m_index = low - m_list.data();
                typename SIMDTraits::value_type lhs = SIMDTraits::init(id);
                typename SIMDTraits::value_type rhs = SIMDTraits::load(low);
                typename SIMDTraits::mask_type le_mask = ~SIMDTraits::gt(lhs, rhs);
                int offset = __builtin_ctz(le_mask) / 4;
                m_index2 = m_index + offset;
                return m_list[m_index2];
            }
            prev = index;
            index += step;
            step *= 2;
        }
        m_index2 = index;

        // fallback to linear search
        for (; m_index2 < m_list.size(); m_index2++) {
            if (m_list[m_index2] >= id) {
                return m_list[m_index2];
            }
        }
        return kInvalidDocId;
    }
};

/**
 *  \tparam Seek1 algorithm used to seek doc id in the 1st list
 *  \tparam Seek2 algorithm used to seek doc id in the 2nd list
 */
template<typename Seek1, typename Seek2 = Seek1>
struct GallopingIntersection
{
    static void intersection(PostingList const &L1, PostingList const &L2, PostingList &out)
    {
        size_t i1 = 0;
        size_t i2 = 0;
        size_t size1 = L1.size();
        while (i1 < size1) {
            DocId id = L1[i1];
            DocId id2 = Seek2::seek(L2, i2, id);
            if (id2 == kInvalidDocId)
                return;
            if (id != id2)
                id = Seek1::seek(L1, i1, id2);
            if (id == id2) {
                out.push_back(id);
                ++i1;
                ++i2;
            }
        }
    }
};

template<typename SIMDTraits>
struct SimdGallopingIntersection
{
    static void intersection(PostingList const &L1, PostingList const &L2, PostingList &out)
    {
        if (L1.empty())
            return;
        SimdGallopingSeeker<SIMDTraits> s1(L1);
        SimdGallopingSeeker<SIMDTraits> s2(L2);
        DocId id = L1.front();
        DocId id2;
        while ((id2 = s2.seek(id)) != kInvalidDocId) {
            if (id != id2)
                id = s1.seek(id2);
            else {
                out.push_back(id);
                id = s1.next();
                if (id == kInvalidDocId)
                    return;
            }
        }
    }
};

static void binarySearchIntersection(PostingList const &L1, PostingList const &L2, PostingList &out)
{
    size_t i1 = 0;
    size_t i2 = 0;
    size_t size1 = L1.size();
    while (i1 < size1) {
        DocId id = L1[i1];
        DocId id2 = BinarySeek::seek(L2, i2, id);
        if (id2 == kInvalidDocId)
            return;
        if (id != id2)
            id = BinarySeek::seek(L1, i1, id2);
        if (id == id2) {
            out.push_back(id);
            ++i1;
            ++i2;
        }
    }
}

static void simdIntersectionV1(PostingList const &L1, PostingList const &L2, PostingList &out)
{
    size_t size1 = L1.size();
    size_t size2_aligned = L2.size() & (~0xFull);
    size_t i = 0;
    size_t j = 0;
    for (; i < size1; i++) {
        while (L2[j+3] < L1[i]) {
            j += 4;
            if (j >= size2_aligned)
                return;
        }
        __m128i r1 = _mm_set1_epi32(L1[i]);
        __m128i r2 = _mm_loadu_si128((__m128i*)(L2.data() + j));
        __m128i r3 = _mm_cmpeq_epi32(r1, r2);
        uint32_t eq_mask = _mm_movemask_epi8(r3);
        if (eq_mask)
            out.push_back(L1[i]);
    }

    // handle the remained
    size_t size2 = L2.size();
    for (; i < size1; i++) {
        while (L2[j] < L1[i]) {
            j++;
            if (j >= size2)
                return;
        }
        if (L1[i] == L2[j])
            out.push_back(L1[i]);
    }
}

static void avxIntersectionV1(PostingList const &L1, PostingList const &L2, PostingList &out)
{
    size_t size1 = L1.size();
    size_t size2_aligned = L2.size() & (~0x1Full);
    size_t i = 0;
    size_t j = 0;
    for (; i < size1; i++) {
        while (L2[j+7] < L1[i]) {
            j += 8;
            if (j >= size2_aligned)
                return;
        }
        __m256i r1 = _mm256_set1_epi32(L1[i]);
        __m256i r2 = _mm256_loadu_si256((__m256i*)(L2.data() + j));
        __m256i r3 = _mm256_cmpeq_epi32(r1, r2);
        uint32_t eq_mask = _mm256_movemask_epi8(r3);
        if (eq_mask)
            out.push_back(L1[i]);
    }

    // handle the remained
    size_t size2 = L2.size();
    for (; i < size1; i++) {
        while (L2[j] < L1[i]) {
            j++;
            if (j >= size2)
                return;
        }
        if (L1[i] == L2[j])
            out.push_back(L1[i]);
    }
}

typedef void (*IntersectionCallback)(PostingList const &, PostingList const &, PostingList &);

static void printList(const char *desc, PostingList const &list)
{
    printf("[%s] [%zu]", desc, list.size());
    if (list.size() <= 10) {
        for (auto v : list) {
            printf(" %u", v);
        }
    }
    else {
        for (size_t i = 0; i < 5; i++) {
            printf(" %u", list[i]);
        }
        printf(" ...");
        for (size_t i = list.size() - 5; i < list.size(); i++) {
            printf(" %u", list[i]);
        }
    }
    printf("\n");
}

/// \returns average time in microseconds
static double benchmark(const char *desc, PostingList const &L1, PostingList const &L2, PostingList &out, IntersectionCallback intersection, size_t loopCount = 100)
{
    const size_t kLoopCount = loopCount;
    const auto t_begin = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < kLoopCount; i++) {
        out.clear();
        intersection(L1, L2, out);
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    const auto timespanUs = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
    const auto avgTimeUs = (double)timespanUs/kLoopCount;
    printf("[%s] total time: %" PRId64 " us, avg: %lf us\n", desc, timespanUs, avgTimeUs);
    printList("result", out);
    return avgTimeUs;
}

// Generates the following sequence until reaching the limit:
// 1 * 10^0, 2 * 10^0, ... , 9 * 10^0,
// 1 * 10^1, 2 * 10^1, ... , 9 * 10^1,
// 1 * 10^2, 2 * 10^2, ... , 9 * 10^2,
// ...
struct ExpLevelGenerator
{
public:
    ExpLevelGenerator(size_t max, size_t initialExp = 0)
    : m_max(max)
    , m_scalar(1)
    {
        m_factor = 1;
        for (size_t lv = 0; lv < initialExp; lv++) {
            m_factor *= 10;
        }
    }
    /// \returns next index, or zero if finished
    size_t next()
    {
        size_t index = m_factor * m_scalar;
        if (m_scalar == 9) {
            m_scalar = 1;
            m_factor *= 10;
        }
        else {
            m_scalar++;
        }
        return index > m_max ? 0 : index;
    }
private:
    size_t m_max;
    size_t m_scalar; // in range [1,9]
    size_t m_factor;
};

static void diagram(const char *desc, IntersectionCallback intersection, DocId maxId)
{
    const DocId kMaxId = maxId;
    PostingList posting_1;
    PostingList posting_2;
    PostingList out;
    size_t initialExp = log(maxId)/log(10) - 1;
    printf("maxId: %u, exp: %zu\n", kMaxId, initialExp);
    ExpLevelGenerator i1(kMaxId, initialExp);
    while (DocId d1 = i1.next()) {
        posting_1.clear();
        generateList(kMaxId, d1, posting_1);
        ExpLevelGenerator i2(kMaxId, initialExp);
        while (DocId d2 = i2.next()) {
            posting_2.clear();
            generateList(kMaxId, d2 + 1, posting_2);  // d2 may be equal to d1, and d2+1 makes a different random seed than d2
            printf("[L1:%zu] [L2:%zu]\n", posting_1.size(), posting_2.size());
            double avgTimeUs = benchmark(desc, posting_1, posting_2, out, intersection, 10);
            printf("[DIAG] [%s] %zu %zu %lf\n", desc, posting_1.size(), posting_2.size(), avgTimeUs);
        }
    }
}

int main(int argc, char *argv[])
{
    const DocId kMaxId = 10000000u;
    PostingList posting_1;
    PostingList posting_2;
    PostingList out;
    generateList(kMaxId, 1000, posting_1);
    generateList(kMaxId, 1000000, posting_2);
    printf("L1: %zu, L2: %zu\n", posting_1.size(), posting_2.size());
    printList("L1", posting_1);
    printList("L2", posting_2);

    typedef GallopingSeek<BinarySeek> NormalGallopingSeek;
    typedef GallopingSeek<SimpleBinarySeek> SimpleGallopingSeek;
    IntersectionCallback gallopingIntersection = GallopingIntersection<BinarySeek, NormalGallopingSeek>::intersection;
    IntersectionCallback gallopingIntersection2 = GallopingIntersection<NormalGallopingSeek>::intersection;
    IntersectionCallback gallopingIntersection3 = GallopingIntersection<SimpleGallopingSeek>::intersection;
    IntersectionCallback sse4GallopingIntersection = SimdGallopingIntersection<SSE4>::intersection;
    IntersectionCallback avx2GallopingIntersection = SimdGallopingIntersection<AVX2>::intersection;

    printf("\n[shorter list first]\n");
    printf("========\n");
    benchmark("linear", posting_1, posting_2, out, linearIntersection);
    printf("--------\n");
    benchmark("binary", posting_1, posting_2, out, binarySearchIntersection);
    printf("--------\n");
    benchmark("gallop", posting_1, posting_2, out, gallopingIntersection);
    printf("--------\n");
    benchmark("gallop2", posting_1, posting_2, out, gallopingIntersection2);
    printf("--------\n");
    benchmark("gallop3", posting_1, posting_2, out, gallopingIntersection3);
    printf("--------\n");
    benchmark("gallop_sse4", posting_1, posting_2, out, sse4GallopingIntersection);
    printf("--------\n");
    benchmark("gallop_avx2", posting_1, posting_2, out, avx2GallopingIntersection);
    printf("--------\n");
    benchmark("simdV1", posting_1, posting_2, out, simdIntersectionV1);
    printf("--------\n");
    benchmark("avxV1", posting_1, posting_2, out, avxIntersectionV1);

    printf("\n[longer list first]\n");
    printf("========\n");
    benchmark("linear", posting_2, posting_1, out, linearIntersection);
    printf("--------\n");
    benchmark("binary", posting_2, posting_1, out, binarySearchIntersection);
    printf("--------\n");
    benchmark("gallop", posting_2, posting_1, out, gallopingIntersection);
    printf("--------\n");
    benchmark("gallop2", posting_2, posting_1, out, gallopingIntersection2);
    printf("--------\n");
    benchmark("gallop3", posting_2, posting_1, out, gallopingIntersection3);
    printf("--------\n");
    benchmark("gallop_sse4", posting_2, posting_1, out, sse4GallopingIntersection);
    printf("--------\n");
    benchmark("gallop_avx2", posting_2, posting_1, out, avx2GallopingIntersection);
    printf("--------\n");
    benchmark("simdV1", posting_2, posting_1, out, simdIntersectionV1);
    printf("--------\n");
    benchmark("avx2V1", posting_2, posting_1, out, avxIntersectionV1);

    posting_1.clear();
    posting_2.clear();
    generateList(kMaxId, 10001, posting_1);
    generateList(kMaxId, 10000, posting_2);
    printf("\n[similar length 10^4]\n");
    printf("========\n");
    benchmark("linear", posting_1, posting_2, out, linearIntersection);
    printf("--------\n");
    benchmark("binary", posting_1, posting_2, out, binarySearchIntersection);
    printf("--------\n");
    benchmark("gallop", posting_1, posting_2, out, gallopingIntersection);
    printf("--------\n");
    benchmark("gallop2", posting_1, posting_2, out, gallopingIntersection2);
    printf("--------\n");
    benchmark("gallop3", posting_1, posting_2, out, gallopingIntersection3);
    printf("--------\n");
    benchmark("gallop_sse4", posting_1, posting_2, out, sse4GallopingIntersection);
    printf("--------\n");
    benchmark("gallop_avx2", posting_1, posting_2, out, avx2GallopingIntersection);
    printf("--------\n");
    benchmark("simdV1", posting_1, posting_2, out, simdIntersectionV1);
    printf("--------\n");
    benchmark("avxV1", posting_1, posting_2, out, avxIntersectionV1);

    posting_1.clear();
    posting_2.clear();
    generateList(kMaxId, 100001, posting_1);
    generateList(kMaxId, 100000, posting_2);
    printf("\n[similar length 10^5]\n");
    printf("========\n");
    benchmark("linear", posting_1, posting_2, out, linearIntersection);
    printf("--------\n");
    benchmark("binary", posting_1, posting_2, out, binarySearchIntersection);
    printf("--------\n");
    benchmark("gallop", posting_1, posting_2, out, gallopingIntersection);
    printf("--------\n");
    benchmark("gallop2", posting_1, posting_2, out, gallopingIntersection2);
    printf("--------\n");
    benchmark("gallop3", posting_1, posting_2, out, gallopingIntersection3);
    printf("--------\n");
    benchmark("gallop_sse4", posting_1, posting_2, out, sse4GallopingIntersection);
    printf("--------\n");
    benchmark("gallop_avx2", posting_1, posting_2, out, avx2GallopingIntersection);
    printf("--------\n");
    benchmark("simdV1", posting_1, posting_2, out, simdIntersectionV1);
    printf("--------\n");
    benchmark("avxV1", posting_1, posting_2, out, avxIntersectionV1);

    return 0;
}
