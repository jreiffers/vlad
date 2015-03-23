/**
 * Copyright (c) 2015 Johannes Reifferscheid
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef VLAD_UTF8_HPP__
#define VLAD_UTF8_HPP__

#include <string>
#include <vector>

namespace utf8 {

inline int bytecount(uint32_t codepoint) {
	if (codepoint < (1u << 7))
		return 1;
	if (codepoint < (1u << 11))
		return 2;
	if (codepoint < (1u << 16))
		return 3;
	return 4;
}

template<typename Iter>
void write(Iter iter, uint32_t codepoint) {
	if (codepoint < (1u << 7)) {
		*iter = (uint8_t)codepoint;
	} else if (codepoint < (1u << 11)) {
		*iter     = 0b1100'0000 | (uint8_t)((codepoint >> 6));
		*(++iter) = 0b1000'0000 | (uint8_t)((codepoint >> 0) & 0x63);
	} else if (codepoint < (1u << 16)) {
		*iter     = 0b1110'0000 | (uint8_t)((codepoint >> 12));
		*(++iter) = 0b1000'0000 | (uint8_t)((codepoint >> 6) & 0x63);
		*(++iter) = 0b1000'0000 | (uint8_t)((codepoint >> 0) & 0x63);
	} else {
		*iter     = 0b1111'0000 | (uint8_t)((codepoint >> 18));
		*(++iter) = 0b1000'0000 | (uint8_t)((codepoint >> 12) & 0x63);
		*(++iter) = 0b1000'0000 | (uint8_t)((codepoint >> 6) & 0x63);
		*(++iter) = 0b1000'0000 | (uint8_t)((codepoint >> 0) & 0x63);
	}
}

// FIXME this doesn't handle resynchronisation correctly (doesn't skip 0b10xxxxxx bytes at the beginning of a code point)
template<typename Iter>
uint32_t next_codepoint(Iter& iter, const Iter end) {
	// the 4 most significant bits of the first byte of a UTF-8 code point encode the length of the 
	// code point:
	//  0xxx -> 1 byte  (0)
	//  110x -> 2 bytes (1)
	//  1110 -> 3 bytes (2)
	//  1111 -> 4 bytes (3)
	// 
	// this can be encoded in the bit string
	//  11 10 01 01 (0...)
	const uint32_t magic = 0xE5000000;

	uint8_t start = (uint8_t) *iter;
	++iter;
	int length = 1 + ((magic >> (2*(start >> 4))) & 3);

	if (length == 1)
		return start;

	int bits_in_start = 7 - length;
	uint32_t result = start & ((1 << bits_in_start) - 1);

	for (int i = 1; i < length && iter != end; ++i) {
		uint8_t chr = (uint8_t) *iter;
		if ((chr >> 6) != 2)
			break;
		result <<= 6;
		result |= chr & 0x3F;
		++iter;
	}

	return result;
}

template <typename Iter, typename OutIter>
void to_utf32(Iter begin, Iter end, OutIter out) {
	while (begin != end)
		*out++ = next_codepoint(begin, end);
}

} // namespace utf8

#endif
