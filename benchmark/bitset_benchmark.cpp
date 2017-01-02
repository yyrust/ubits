#include <ubits/bitset.h>
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

namespace ubits {

static const uint64_t kBitMask64[64] = {
    0x0000000000000001ull, 0x0000000000000002ull, 0x0000000000000004ull, 0x0000000000000008ull,
    0x0000000000000010ull, 0x0000000000000020ull, 0x0000000000000040ull, 0x0000000000000080ull,
    0x0000000000000100ull, 0x0000000000000200ull, 0x0000000000000400ull, 0x0000000000000800ull,
    0x0000000000001000ull, 0x0000000000002000ull, 0x0000000000004000ull, 0x0000000000008000ull,
    0x0000000000010000ull, 0x0000000000020000ull, 0x0000000000040000ull, 0x0000000000080000ull,
    0x0000000000100000ull, 0x0000000000200000ull, 0x0000000000400000ull, 0x0000000000800000ull,
    0x0000000001000000ull, 0x0000000002000000ull, 0x0000000004000000ull, 0x0000000008000000ull,
    0x0000000010000000ull, 0x0000000020000000ull, 0x0000000040000000ull, 0x0000000080000000ull,
    0x0000000100000000ull, 0x0000000200000000ull, 0x0000000400000000ull, 0x0000000800000000ull,
    0x0000001000000000ull, 0x0000002000000000ull, 0x0000004000000000ull, 0x0000008000000000ull,
    0x0000010000000000ull, 0x0000020000000000ull, 0x0000040000000000ull, 0x0000080000000000ull,
    0x0000100000000000ull, 0x0000200000000000ull, 0x0000400000000000ull, 0x0000800000000000ull,
    0x0001000000000000ull, 0x0002000000000000ull, 0x0004000000000000ull, 0x0008000000000000ull,
    0x0010000000000000ull, 0x0020000000000000ull, 0x0040000000000000ull, 0x0080000000000000ull,
    0x0100000000000000ull, 0x0200000000000000ull, 0x0400000000000000ull, 0x0800000000000000ull,
    0x1000000000000000ull, 0x2000000000000000ull, 0x4000000000000000ull, 0x8000000000000000ull,
};

static const uint32_t kBitMask32[32] = {
    0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u,
    0x00000010u, 0x00000020u, 0x00000040u, 0x00000080u,
    0x00000100u, 0x00000200u, 0x00000400u, 0x00000800u,
    0x00001000u, 0x00002000u, 0x00004000u, 0x00008000u,
    0x00010000u, 0x00020000u, 0x00040000u, 0x00080000u,
    0x00100000u, 0x00200000u, 0x00400000u, 0x00800000u,
    0x01000000u, 0x02000000u, 0x04000000u, 0x08000000u,
    0x10000000u, 0x20000000u, 0x40000000u, 0x80000000u,
};

template<typename Unit>
struct LutMaskPolicy;

template<> struct LutMaskPolicy<uint64_t>
{
    static uint64_t unitMask(size_t bitIndex) { return kBitMask64[bitIndex]; }
};

template<> struct LutMaskPolicy<uint32_t>
{
    static uint32_t unitMask(size_t bitIndex) { return kBitMask32[bitIndex]; }
};

struct NextOperation
{
    template<typename BitsetT>
    size_t op(const BitsetT &bitset, int key)
    {
        return bitset.next(key);
    }
    static const char *name() { return "next"; }
};

struct TestOperation
{
    template<typename BitsetT>
    size_t op(const BitsetT &bitset, int key)
    {
        return bitset.unsafeContains(key);
    }
    static const char *name() { return "test"; }
};

template<typename BitsetT, typename Operation>
void benchmark(const std::vector<int> &keys, const BitsetT &bitset, Operation op, const char *desc)
{
    size_t sum = 0;
    const auto tbegin = std::chrono::high_resolution_clock::now();
    for (auto &k : keys) {
        size_t i = op.op(bitset, k);
        sum += i; // avoid optimization
    }
    const auto tend = std::chrono::high_resolution_clock::now();
    const auto timespan = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count();
    printf("[%s %s] sum: %zd, time: %" PRId64 "\n", desc, op.name(), sum, timespan);
}

template<typename Operation>
static void benchmarkSuite(int numTotalDocs, int numDocs, int numKeys)
{
    printf("total: %d, docs: %d, keys: %d\n", numTotalDocs, numDocs, numKeys);

    std::mt19937 rng(303);

    std::uniform_int_distribution<int> valueDist(0, numTotalDocs-1);
    typedef BitsetTraits<uint32_t> bt32;
    typedef BitsetTraits<uint64_t> bt64;
    typedef BitsetTraits<uint32_t, LutMaskPolicy<uint32_t> > bt32lut;
    typedef BitsetTraits<uint64_t, LutMaskPolicy<uint64_t> > bt64lut;
    Bitset<bt32> bitset32;
    Bitset<bt64> bitset64;
    Bitset<bt32lut> bitset32_lut;
    Bitset<bt64lut> bitset64_lut;
    bitset32.resize(numTotalDocs);
    bitset64.resize(numTotalDocs);
    bitset32_lut.resize(numTotalDocs);
    bitset64_lut.resize(numTotalDocs);
    for (int i = 0; i < numDocs; i++) {
        auto v = valueDist(rng);
        bitset32.unsafeInsert(v);
        bitset64.unsafeInsert(v);
        bitset32_lut.unsafeInsert(v);
        bitset64_lut.unsafeInsert(v);
    }

    std::uniform_int_distribution<int> keyDist(0, numTotalDocs-1);
    std::vector<int> keys(numKeys);
    for (auto &k : keys)
        k = keyDist(rng);

    Operation op;
    benchmark<Bitset<bt64>, Operation>(keys, bitset64, op, "bitset64    ");
    benchmark<Bitset<bt32>, Operation>(keys, bitset32, op, "bitset32    ");
    benchmark<Bitset<bt64lut>, Operation>(keys, bitset64_lut, op, "bitset64_lut");
    benchmark<Bitset<bt32lut>, Operation>(keys, bitset32_lut, op, "bitset32_lut");
}

} // namespace ubits

int main()
{
    using namespace ubits;

    typedef BitsetTraits<uint64_t> bt64;
    Bitset<bt64> bitset64;
    bitset64.resize(1024);
    bitset64.unsafeInsert(3);
    bitset64.unsafeInsert(9);
    bitset64.unsafeInsert(7);
    assert(bitset64.next(3) == 3);
    assert(bitset64.next(4) == 7);
    assert(bitset64.next(7) == 7);
    assert(bitset64.next(8) == 9);
    assert(bitset64.next(9) == 9);

    int kNumKeys = 10000000;
    int kTotalDocCount = 1000000;
    benchmarkSuite<NextOperation>(kTotalDocCount, 500000, kNumKeys);
    benchmarkSuite<NextOperation>(kTotalDocCount, 100000, kNumKeys);
    benchmarkSuite<NextOperation>(kTotalDocCount, 10000, kNumKeys);
    benchmarkSuite<NextOperation>(kTotalDocCount, 1000, kNumKeys);
    benchmarkSuite<NextOperation>(kTotalDocCount, 100, kNumKeys);
    benchmarkSuite<NextOperation>(kTotalDocCount, kTotalDocCount/32, kNumKeys);

    benchmarkSuite<TestOperation>(kTotalDocCount, 500000, kNumKeys);
    benchmarkSuite<TestOperation>(kTotalDocCount, kTotalDocCount/32, kNumKeys);
    benchmarkSuite<TestOperation>(kTotalDocCount, 100, kNumKeys);

    return 0;
}
