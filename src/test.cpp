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
#define BOOST_TEST_MODULE vlad_tests
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <ctime>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <vector>

#define DEBUG
#include <levenshtein.hpp>
#include <trie.hpp>
#include <utf8.hpp>

using namespace vlad;
using namespace vlad::trie;
using namespace vlad::levenshtein;

static uint32_t levenshtein_distance(const std::string& s1, const std::string& s2) {
	const size_t len1 = s1.size();
	const size_t len2 = s2.size();
	std::vector<uint32_t> col(len2 + 1);
	std::vector<uint32_t> prevCol(len2 + 1);

	for (unsigned int i = 0; i < prevCol.size(); i++)
		prevCol[i] = i;

	for (unsigned int i = 0; i < len1; i++) {
		col[0] = i + 1;
		for (unsigned int j = 0; j < len2; j++)
			col[j + 1] = std::min(std::min(1 + col[j], 1 + prevCol[1 + j]),
			                      prevCol[j] + (s1[i] == s2[j] ? 0 : 1));
		col.swap(prevCol);
	}

	return prevCol[len2];
}

// adapted from
// https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance#Optimal_string_alignment_distance
static uint32_t optimal_string_alignment(const std::string& s1, const std::string& s2) {
	std::vector<std::vector<size_t>> d(s1.size() + 1, std::vector<size_t>(s2.size() + 1));

	std::array<std::vector<size_t>, 3> cols;
	cols.fill(std::vector<size_t>(s2.size() + 1));

	for (size_t i = 0; i <= s2.size(); ++i) {
		cols[0][i] = i;
	}

	for (size_t i = 0; i < s1.size(); i++) {
		auto& prev_col = cols[(i + 2) % 3];
		auto& crnt_col = cols[i % 3];
		auto& next_col = cols[(i + 1) % 3];
		next_col[0] = i + 1;

		for (size_t j = 0; j < s2.size(); j++) {
			auto cost = (s1[i] == s2[j]) ? 0 : 1;

			next_col[j + 1] =
			    std::min(std::min(crnt_col[j + 1] + 1, next_col[j] + 1), crnt_col[j] + cost);
			if (i > 0 && j > 0 && (s1[i] == s2[j - 1] && s1[i - 1] == s2[j])) {
				next_col[j + 1] = std::min(next_col[j + 1], prev_col[j - 1] + cost);
			}
		}
	}
	return cols[s1.size() % 3][s2.size()];
}

static void enumerate_strings(int max_length, std::function<void(const std::string&)> callback,
                              const std::string& prefix = "") {
	callback(prefix);
	if (max_length > 0) {
		enumerate_strings(max_length - 1, callback, prefix + "0");
		enumerate_strings(max_length - 1, callback, prefix + "1");
		enumerate_strings(max_length - 1, callback, prefix + "2");
	}
}

#ifdef DEBUG
#if 0
static void print_state(const levenshtein_automaton& a, const levenshtein_matcher<>& m) {
	const auto& positions = a.nfa.states[m.state.id].positions;
	for (const auto& pos : positions) {
		std::cout << "(" << (pos.i + m.state.pos) << ", " << pos.e << (pos.trans ? "t" : "") << ")";
	}
	std::cout << std::endl;
}
#endif
#endif

static std::vector<uint32_t> to_utf32(const std::string& s) {
	std::vector<uint32_t> result;
	::utf8::to_utf32(s.begin(), s.end(), std::back_inserter(result));
	return result;
}

BOOST_AUTO_TEST_SUITE(trie_tests)

template <typename V>
void simple_trie_test(V visitor) {
	word_index index;

	auto insert = [&](const std::string& s, uint32_t val) {
		auto v = to_utf32(s);
		index.insert(v.begin(), v.end(), val);
	};

	visitor(insert);

#ifdef DEBUG
	index.print();
#endif

	levenshtein_automaton a0(0, false);

	visitor([&](const std::string& s, uint32_t val) {
		size_t call_count = 0;
		uint32_t value = 0;
		levenshtein_matcher<> m0(a0, s);
		index.find(m0, [&](vector_wrapper val) {
			++call_count;
			BOOST_REQUIRE(val->size() == 1);
			value = (*val)[0];
		});

		BOOST_CHECK_MESSAGE(call_count == 1, s << ": call_count = " << call_count);
		BOOST_CHECK_MESSAGE(value == val, s << ": value = " << value);
	});
}

bool verify_fuzz(const word_index& index, const levenshtein_automaton& a,
                 const std::vector<uint32_t>& key, const std::vector<uint32_t>& docs) {
	levenshtein_matcher<profiles::utf32::naive> m(a, key);

	bool result = true;
	size_t call_count = 0;
	index.find(m, [&](vector_wrapper val) {
		++call_count;
		result &= *val == docs;
	});

	if (!result || call_count != 1) {
		std::cout << "fuzz test failed for key {";
		for (auto v : key)
			std::cout << v << ", ";
		std::cout << "} with call_count = " << call_count;
		std::cout << std::endl;
	}

	return result && call_count == 1;
}

template <typename V>
void u16_trie_test(V visitor) {
	word_index index;

	auto inserter = [&](const std::vector<uint32_t>& v, uint32_t val) {
		index.insert(v.begin(), v.end(), val);
	};

	levenshtein_automaton a(0, false);
	auto verifier = [&](const std::vector<uint32_t>& v, uint32_t val) {
		BOOST_CHECK(verify_fuzz(index, a, v, { val }));
	};

	visitor(inserter);
	visitor(verifier);
}

BOOST_AUTO_TEST_CASE(utf8) {
	/*
	confirms that this file is encoded correctly
	*/
	std::string s = "ﬀ";
	auto iter = s.begin();
	uint32_t codepoint = ::utf8::next_codepoint(iter, s.end());

	BOOST_CHECK(codepoint == 0xFB00);
}

BOOST_AUTO_TEST_CASE(empty) {
	/*
	confirms that the empty string is handled correctly.
	*/
	simple_trie_test([](auto cb) { cb("", 1); });
}

BOOST_AUTO_TEST_CASE(prefix) {
	/*
	confirms that inserting a new value whose key is a prefix of a previously inserted
	value works.
	*/

	simple_trie_test([](auto cb) {
		cb("hello world", 1);
		cb("hello", 2);
	});
}

BOOST_AUTO_TEST_CASE(compound) {
	/*
	confirms that inserting values with code points >= 32768 works
	*/

	simple_trie_test([](auto cb) {
		cb("aﬁbﬂc", 1);
		cb("aﬂbﬂc", 2);
		cb("aﬂﬂﬂc", 3);
	});
}

BOOST_AUTO_TEST_CASE(long_tail) {
	/*
	confirms that splitting a leaf works when the new key is very long
	*/
	simple_trie_test([](auto cb) {
		cb("13", 1);
		cb("123456789012345", 2);
	});
}

BOOST_AUTO_TEST_CASE(full_tail_prefix) {
	/*
	confirms that splitting a leaf works when the splitted leaf has a full tail and is a prefix of
	the new key
	*/
	simple_trie_test([](auto cb) {
		cb("12345678901", 1);
		cb("123456789012345", 2);
	});
}

BOOST_AUTO_TEST_CASE(full_tail_same) {
	/*
	confirms that inserting a value works when there's already an entry for the key and the entry's
	leaf has a full tail.
	*/
	std::string str("012345678901234567890");
	auto codepoints = to_utf32(str);

	for (size_t len = 2; len < str.size(); ++len) {
		word_counter index;

		index.insert(codepoints.begin(), codepoints.begin() + len, { 1, 0 });
		index.insert(codepoints.begin(), codepoints.begin() + len, { 2, 1 });
		BOOST_CHECK_EQUAL(1, index.size());

#ifdef DEBUG
		index.print();
#endif

		levenshtein_automaton a0(0, false);

		size_t call_count = 0;
		word_counter::value_type value;
		levenshtein_matcher<> m0(a0, std::string(str.begin(), str.begin() + len));
		index.find(m0, [&](const word_counter::value_type& val) {
			++call_count;
			value = val;
		});

		BOOST_CHECK_EQUAL(1, call_count);
		BOOST_CHECK_EQUAL(3, value.count);
		BOOST_CHECK_EQUAL(1, value.max_document_id);
	}
}

BOOST_AUTO_TEST_CASE(wide) {
	/*
	confirms that inner nodes with more than 4 children work.
	*/

	std::vector<std::string> strs{ "aaa", "aba", "aca", "ada", "aea", "afa", "aga" };
	simple_trie_test([&](auto cb) {
		uint32_t n = 0;
		for (const auto& str : strs)
			cb(str, ++n);
	});
}

bool fuzz_test(size_t seed, const std::vector<bool> mask, int maxlen = -1, bool print = false) {
	/*
	Generates random test data. If the test data triggers a bug, finds a minimal subset of the data
	that still triggers the bug and outputs code for a test case.
	*/

	std::map<std::vector<uint32_t>, std::vector<uint32_t>> entries;

	std::uniform_int_distribution<> len(0, 15);
	std::uniform_int_distribution<> chars(1, 100000);

	word_index index;

	std::mt19937 rd;
	rd.seed(seed);
	std::vector<uint32_t> v;

	if (print) {
		std::cout << "BOOST_AUTO_TEST_CASE(fuzz_test_" << seed << ") {" << std::endl;
		std::cout << "\tauto visitor = [](auto cb) {" << std::endl;
	}

	size_t size = 0;
	for (uint32_t i = 0; i < mask.size(); ++i) {
		v.clear();

		int l = len(rd);
		for (int j = 0; j < l; ++j) {
			int chr = chars(rd);
			if (maxlen > 0 && j >= maxlen - 1)
				continue;

			v.push_back(chr);
		}

		if (mask[i]) {
			++size;

			if (print) {
				std::cout << "\t\tcb({";
				std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, ", "));
				std::cout << "}, " << i << ");" << std::endl;
			}

			std::vector<uint32_t>& docs = entries[v];
			docs.push_back(i);

			index.insert(v.begin(), v.end(), i);
		}
	}

	if (print) {
		std::cout << "\t};" << std::endl;
		std::cout << "\tu16_trie_test(visitor);" << std::endl;
		std::cout << "}" << std::endl;
	}

	// verify entries
	rd.seed(seed);
	bool result = true;

	levenshtein_automaton a0(0, false);
	for (const auto& e : entries) {
		bool this_elem = verify_fuzz(index, a0, e.first, e.second);
		result &= this_elem;
	}

	return result;
}

BOOST_AUTO_TEST_CASE(fuzz) {
	size_t seed = time(nullptr);

	std::cout << "seed: " << seed << std::endl;
	std::vector<bool> mask(1024, true);

	if (!fuzz_test(seed, mask)) {
		for (size_t step_size = mask.size() / 2; step_size > 0; step_size >>= 1) {
			std::cout << "step size: " << step_size << std::endl;
			for (size_t pos = 0; pos < mask.size(); pos += step_size) {
				if (!mask[pos])
					continue;
				for (size_t i = 0; i < step_size; ++i)
					mask[pos + i] = false;

				// if the test doesn't still fail, re-enable this chunk
				if (fuzz_test(seed, mask)) {
					for (size_t i = 0; i < step_size; ++i)
						mask[pos + i] = true;
				}
			}
		}

		int max_len = 20;
		while (max_len > 0 && !fuzz_test(seed, mask, max_len, false))
			--max_len;

		BOOST_CHECK(fuzz_test(seed, mask, max_len + 1, true));
	} else {
		BOOST_CHECK(true);
	}
}

BOOST_AUTO_TEST_CASE(basic) {
	word_index index;

	// insert strings
	uint32_t n = 0;
	enumerate_strings(5, [&](const std::string& s1) {
		BOOST_TEST_CHECKPOINT("inserting " << s1);
		auto u32 = to_utf32(s1);
		index.insert(u32.begin(), u32.end(), ++n);
	});

	uint32_t dist = n;
	enumerate_strings(5, [&](const std::string& s1) {
		BOOST_TEST_CHECKPOINT("inserting " << s1);
		auto u32 = to_utf32(s1);
		index.insert(u32.begin(), u32.end(), ++n);
	});

	levenshtein_automaton a0(0, false);
	// check that strings are in the index
	n = 0;
	enumerate_strings(5, [&](const std::string& s1) {
		size_t call_count = 0;
		uint32_t value = 0;

		levenshtein_matcher<> m0(a0, s1);
		index.find(m0, [&](vector_wrapper val) {
			++call_count;
			BOOST_REQUIRE(val->size() == 2);
			value = (*val)[0];
			BOOST_CHECK((*val)[1] - (*val)[0] == dist);
		});

		++n;
		BOOST_CHECK_MESSAGE(call_count == 1, s1 << ": call_count was " << call_count);
		BOOST_CHECK_MESSAGE(value == n, s1 << ": value = " << value << " != " << n);
	});
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(str_mask)

BOOST_AUTO_TEST_CASE(differential_str_mask) {
	/*
	 * Confirms that the different str mask implementations do the same thing.
	 */

	enumerate_strings(6, [](const std::string& s) {
		profiles::ansi::precomputed<> ansi_pre(s, 2);
		profiles::utf32::naive utf32_naive(to_utf32(s), 2);

#ifdef __SSE4_2__
		profiles::ansi::sse ansi_sse(s, 2);
#endif

		for (int i = 0; i < 6; ++i) {
			for (char c = '0'; c <= '3'; ++c) {
				BOOST_CHECK_EQUAL(ansi_pre(i, c), utf32_naive(i, c));
#ifdef __SSE4_2__
				BOOST_CHECK_EQUAL(ansi_pre(i, c), ansi_sse(i, c));
#endif
			}
		}
	});
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(levenshtein)

BOOST_AUTO_TEST_CASE(no_trans) {
	/*
	 * Confirms that levenshtein-automata with max_distance = 0, 1, 2 and no transpositions
	 * accept the right words.
	 */

	levenshtein_automaton a0(0, false);
	levenshtein_automaton a1(1, false);
	levenshtein_automaton a2(2, false);

	enumerate_strings(5, [&](const std::string& s1) {
		enumerate_strings(5, [&](const std::string& s2) {
			int dist = levenshtein_distance(s1, s2);

			levenshtein_matcher<> m0(a0, s1);
			levenshtein_matcher<> m1(a1, s1);
			levenshtein_matcher<> m2(a2, s1);

			for (char c : s2) {
				m0.next(c);
				m1.next(c);
				m2.next(c);
			}

			BOOST_CHECK_MESSAGE((dist == 0) == m0.accepted(), s1 << " vs. " << s2
			                                                     << " not accepted by m0");
			BOOST_CHECK_MESSAGE((dist <= 1) == m1.accepted(), s1 << " vs. " << s2
			                                                     << " not accepted by m1");
			BOOST_CHECK_MESSAGE((dist <= 2) == m2.accepted(), s1 << " vs. " << s2
			                                                     << " not accepted by m2");
		});
	});
}

BOOST_AUTO_TEST_CASE(trans) {
	/*
	 * Confirms that levenshtein-automata with max_distance = 0, 1, 2 and transpositions
	 * accept the right words.
	 */

	levenshtein_automaton a0(0, true);
	levenshtein_automaton a1(1, true);
	levenshtein_automaton a2(2, true);

	enumerate_strings(5, [&](const std::string& s1) {
		enumerate_strings(5, [&](const std::string& s2) {
			int dist = optimal_string_alignment(s1, s2);

			levenshtein_matcher<> m0(a0, s1);
			levenshtein_matcher<> m1(a1, s1);
			levenshtein_matcher<> m2(a2, s1);

			for (char c : s2) {
				m0.next(c);
				m1.next(c);
				m2.next(c);
			}

			BOOST_CHECK_MESSAGE((dist == 0) == m0.accepted(), s1 << " vs. " << s2
			                                                     << (m0.accepted() ? "" : " not")
			                                                     << " accepted by m0");
			BOOST_CHECK_MESSAGE((dist <= 1) == m1.accepted(), s1 << " vs. " << s2
			                                                     << (m1.accepted() ? "" : " not")
			                                                     << " accepted by m1");
			BOOST_CHECK_MESSAGE((dist <= 2) == m2.accepted(), s1 << " vs. " << s2
			                                                     << (m2.accepted() ? "" : " not")
			                                                     << " accepted by m2");
		});
	});
}

BOOST_AUTO_TEST_SUITE_END()
