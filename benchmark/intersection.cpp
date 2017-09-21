#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <smmintrin.h>
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
    size_t j = 0;
    for (size_t i = 0; i < size1; i++) {
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
    // TODO: handle the remained
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

static void benchmark(const char *desc, PostingList const &L1, PostingList const &L2, PostingList &out, IntersectionCallback intersection)
{
    const size_t kLoopCount = 1000;
    const auto t_begin = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < kLoopCount; i++) {
        out.clear();
        intersection(L1, L2, out);
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    const auto timespanUs = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
    printf("[%s] total time: %" PRId64 " us, avg: %lf us\n", desc, timespanUs, (double)timespanUs/kLoopCount);
    printList("result", out);
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
    benchmark("simdV1", posting_1, posting_2, out, simdIntersectionV1);

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
    benchmark("simdV1", posting_2, posting_1, out, simdIntersectionV1);

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
    benchmark("simdV1", posting_1, posting_2, out, simdIntersectionV1);

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
    benchmark("simdV1", posting_1, posting_2, out, simdIntersectionV1);

    return 0;
}
