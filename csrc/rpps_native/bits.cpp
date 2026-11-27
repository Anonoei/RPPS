// Bit-level kernels on hard bits (one bit a byte, 0 or 1; the reference loops are in rpps/math/bits.py)
//
// bit_distance: the Hamming distance to a word of up to 64 bits at every position of a bit stream, from a shift register
// and a popcount a bit: sync words, flags, markers, with errors and either polarity (L - distance).
// bit_unstuff: bit destuffing (HDLC: a 0 after `stuff` ones is dropped, `stuff` + 1 ones abort) over many spans of one
// stream in a call, each span's bits packed into bytes least significant bit first.
#include "common.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rpps {

namespace {

py::array_t<int16_t> bit_distance(carray<uint8_t> bits_a, carray<uint8_t> word_a) {
    const long long n = bits_a.size(), L = word_a.size();
    if (L < 1 || L > 64) throw std::invalid_argument("bit_distance: a word of 1 to 64 bits");
    const long long m = n >= L ? n - L + 1 : 0;
    auto out = empty<int16_t>(static_cast<std::size_t>(m));
    const uint8_t *b = bits_a.data(), *w = word_a.data();
    int16_t *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        uint64_t W = 0;
        for (long long i = 0; i < L; i++) W = (W << 1) | (w[i] & 1u);
        const uint64_t mask = L == 64 ? ~0ULL : ((1ULL << L) - 1);
        uint64_t reg = 0;
        for (long long k = 0; k < n; k++) {
            reg = ((reg << 1) | (b[k] & 1u)) & mask; // the newest bit lowest: the word's last bit
            if (k >= L - 1) o[k - L + 1] = static_cast<int16_t>(popcount64(reg ^ W));
        }
    }
    return out;
}

// For spans [starts[i], ends[i]) of bits: (status per span: 0 whole bytes, 1 aborted (stuff + 1 ones), 2 not a whole
// number of bytes; offsets (spans + 1) into the bytes; bytes: each good span's destuffed bits, least significant first)
py::tuple bit_unstuff(carray<uint8_t> bits_a, carray<int64_t> starts_a, carray<int64_t> ends_a, int stuff) {
    const long long n = bits_a.size(), S = starts_a.size();
    if (ends_a.size() != S) throw std::invalid_argument("bit_unstuff: as many ends as starts");
    if (stuff < 1) throw std::invalid_argument("bit_unstuff: stuff >= 1");
    const uint8_t *b = bits_a.data();
    const int64_t *st = starts_a.data(), *en = ends_a.data();
    auto status = empty<int8_t>(static_cast<std::size_t>(S));
    auto offsets = empty<int64_t>(static_cast<std::size_t>(S + 1));
    int8_t *s_o = status.mutable_data();
    int64_t *off = offsets.mutable_data();
    std::vector<uint8_t> bytes;
    {
        py::gil_scoped_release nogil;
        off[0] = 0;
        for (long long i = 0; i < S; i++) {
            const long long a = std::max<long long>(st[i], 0), e = std::min<long long>(en[i], n);
            const std::size_t first = bytes.size();
            uint8_t acc = 0;
            int nb = 0, ones = 0, bad = 0;
            for (long long k = a; k < e; k++) {
                const uint8_t bit = b[k] & 1u;
                if (ones == stuff) {
                    ones = 0;
                    if (!bit) continue; // the stuffed zero
                    bad = 1; // more than `stuff` ones inside a span: an abort
                    break;
                }
                acc |= static_cast<uint8_t>(bit << nb);
                if (++nb == 8) { bytes.push_back(acc); acc = 0; nb = 0; }
                ones = bit ? ones + 1 : 0;
            }
            if (!bad && nb) bad = 2;
            if (bad) bytes.resize(first);
            s_o[i] = static_cast<int8_t>(bad);
            off[i + 1] = static_cast<int64_t>(bytes.size());
        }
    }
    py::array_t<uint8_t> out(static_cast<py::ssize_t>(bytes.size()));
    if (!bytes.empty()) std::copy(bytes.begin(), bytes.end(), out.mutable_data());
    return py::make_tuple(status, offsets, out);
}

} // namespace

void register_bits(py::module_ &m) {
    m.def("bit_distance", &bit_distance,
          "bit_distance(bits uint8 (0/1), word uint8 (0/1, 1..64 bits)) -> int16: the Hamming distance to the word at every position",
          py::arg("bits"), py::arg("word"));
    m.def("bit_unstuff", &bit_unstuff,
          "bit_unstuff(bits uint8 (0/1), starts int64, ends int64, stuff) -> (status int8 per span: 0 ok, 1 aborted, 2 not whole bytes; "
          "offsets int64 (spans + 1); bytes uint8: each good span destuffed, least significant bit first)",
          py::arg("bits"), py::arg("starts"), py::arg("ends"), py::arg("stuff"));
}

} // namespace rpps
