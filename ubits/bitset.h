#ifndef UBITS_BITSET_H
#define UBITS_BITSET_H

#include <cstddef>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

namespace ubits {

#ifndef UBITS_STATIC_ASSERT
#define UBITS_STATIC_ASSERT(PRED, MSG) typedef char static_assert__##MSG[(PRED) ? 1 : -1] __attribute__((unused));
#endif

inline size_t lsb(uint32_t value)
{
    return __builtin_ctz(value);
}

inline size_t lsb(uint64_t value)
{
    return __builtin_ctzl(value);
}

template<typename Unit>
struct ShiftMaskPolicy
{
    static Unit unitMask(size_t bitIndex) { return Unit(1) << bitIndex; }
};

template<typename Unit, typename MaskPolicy = ShiftMaskPolicy<Unit> >
class BitsetTraits
{
public:
    typedef Unit unit_type;
    typedef std::size_t size_type;
    typedef std::size_t value_type;
    static const size_type kUnitBits = sizeof(unit_type) * CHAR_BIT;
    static const value_type kInvalidValue = ~value_type();

    UBITS_STATIC_ASSERT(((unit_type(0)-1)) > 0, unit_type_should_be_unsigned);

public:
    inline static size_type unitCount(size_type nBits)
    {
        return (nBits + kUnitBits - 1) / kUnitBits;
    }
    inline static size_type unitIndex(value_type value)
    {
        return value / kUnitBits;
    }
    inline static unit_type unitMask(value_type value)
    {
        return MaskPolicy::unitMask(value % kUnitBits);
    }
};

// read-only bitset
template <typename BitsetTraitsT>
class BitsetView
{
public:
    typedef BitsetTraitsT traits_type;
    typedef typename traits_type::unit_type unit_type;
    typedef typename traits_type::size_type size_type;
    typedef typename traits_type::value_type value_type;
public:
    BitsetView(const unit_type *data, size_type sizeInBits)
        : m_data(data)
        , m_sizeInBits(sizeInBits)
    {
    }

    BitsetView()
        : m_data(0)
        , m_sizeInBits(0)
    {
    }

    void reset(const unit_type *data, size_type sizeInBits)
    {
        m_data = data;
        m_sizeInBits = sizeInBits;
    }

public:

    value_type invalidValue() const { return traits_type::kInvalidValue; }

    size_type capacity() const { return m_sizeInBits; }
    const uint64_t *data() const { return m_data; }

    bool contains(value_type value) const
    {
        return (value < m_sizeInBits) ? unsafeContains(value) : false;
    }

    /// \return the minimal element not less than \c value, or \c invalidValue() if not found
    value_type next(value_type value) const
    {
        return (value < m_sizeInBits) ? unsafeNext(value) : invalidValue();
    }

    /// \pre value < m_sizeInBits
    inline bool unsafeContains(value_type value) const
    {
        size_type nUnitIndex = traits_type::unitIndex(value);
        return m_data[nUnitIndex] & traits_type::unitMask(value);
    }

    /// \return the minimal element not less than \c value, or \c invalidValue() if not found
    /// \pre value < m_sizeInBits
    value_type unsafeNext(value_type value) const
    {
        size_type nUnitIndex = traits_type::unitIndex(value);
        unit_type unit = m_data[nUnitIndex];
        unit_type remain = unit & ~(traits_type::unitMask(value)-1);
        if (remain > 0) {
            return lsb(remain) + nUnitIndex * traits_type::kUnitBits;
        }

        size_type nUnitCount = traits_type::unitCount(m_sizeInBits);
        for (nUnitIndex++; nUnitIndex < nUnitCount; nUnitIndex++) {
            unit = m_data[nUnitIndex];
            if (unit > 0) {
                return lsb(unit) + nUnitIndex * traits_type::kUnitBits;
            }
        }
        return invalidValue();
    }
private:
    // ALWAYS KEEP DATA MEMBER LAYOUT THE SAME AS Bitset!!!! 
    const unit_type *m_data;
    size_t m_sizeInBits;
};

template <typename BitsetTraitsT>
class Bitset
{
public:
    typedef BitsetTraitsT traits_type;
    typedef typename traits_type::unit_type unit_type;
    typedef typename traits_type::size_type size_type;
    typedef typename traits_type::value_type value_type;
    typedef BitsetView<traits_type> view_type;
public:
    Bitset()
        : m_data(NULL)
        , m_sizeInBits(0)
    {
    }

    Bitset(size_type sizeInBits)
        : m_data(NULL)
        , m_sizeInBits(0)
    {
        resize(sizeInBits);
    }

    ~Bitset()
    {
        std::free(m_data);
    }

public:

    size_type capacity() const { return m_sizeInBits; }
    const unit_type *data() const { return m_data; }

    inline void insert(value_type value)
    {
        if (value >= m_sizeInBits)
            resize(value + 1);
        unsafeInsert(value);
    }

    inline void erase(value_type value)
    {
        if (value < m_sizeInBits)
            unsafeErase(value);
    }

    void resize(size_type numBits)
    {
        if (numBits > m_sizeInBits) {
            size_type sizeInUnits = traits_type::unitCount(m_sizeInBits);
            size_type newSizeInUnits = traits_type::unitCount(numBits);
            if (newSizeInUnits > sizeInUnits) {
                m_data = reinterpret_cast<unit_type *>(std::realloc(m_data, newSizeInUnits * sizeof(unit_type)));
                memset(m_data + sizeInUnits, 0, (newSizeInUnits - sizeInUnits) * sizeof(unit_type));
            }
        }
        m_sizeInBits = numBits;
    }

    /// \pre value < m_sizeInBits
    inline void unsafeInsert(value_type value)
    {
        size_type nUnitIndex = traits_type::unitIndex(value);
        m_data[nUnitIndex] |= traits_type::unitMask(value);
    }

    /// \pre value < m_sizeInBits
    inline void unsafeErase(value_type value)
    {
        size_type nUnitIndex = unitIndex(value);
        m_data[nUnitIndex] &= ~unitMask(value);
    }

public: // forward read-only operations to BitsetView

    inline const view_type *asView() const
    {
        return reinterpret_cast<const view_type *>(this);
    }

    inline value_type invalidValue() const { return asView()->invalidValue(); }

    inline bool contains(value_type value) const
    {
        return asView()->contains(value);
    }

    /// \copydoc BitsetView::next
    inline value_type next(value_type value) const
    {
        return asView()->next(value);
    }

    /// \copydoc BitsetView::unsafeContains
    inline bool unsafeContains(value_type value) const
    {
        return asView()->unsafeContains(value);
    }

    /// \copydoc BitsetView::unsafeNext
    inline value_type unsafeNext(value_type value) const
    {
        return asView()->unsafeNext(value);
    }

private:
    // ALWAYS KEEP DATA MEMBER LAYOUT THE SAME AS BitsetView!!!! 
    unit_type *m_data;
    size_t m_sizeInBits;

private:
    // non-copyable
    Bitset(const Bitset &);
    Bitset &operator=(const Bitset &);
};

} // namespace ubits

#endif // UBITS_BITSET_H
