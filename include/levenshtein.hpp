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
#ifndef VLAD_LEVENSHTEIN_HPP__
#define VLAD_LEVENSHTEIN_HPP__

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __SSE4_2__
#include <nmmintrin.h>
#include <smmintrin.h>
#endif

#include "utf8.hpp"

namespace vlad {

namespace levenshtein {

/**
 * \brief Functors for computing string bit masks using strings with various encodings.
 *
 * Bit `n` in the bit mask encodes whether character `n` in a given string equals a given character.
 * For instance, the bit mask for the string `hello` and the character `l` is 12 (0b1100). The bit
 * mask is 2*`max_distance`+1 bits long, where `max_distance` is the maximum distance of the
 * levenshtein automaton. Bits that are beyond the end of the string are set to zero.
 *
 * There are functors for ANSI, UTF-8 and UTF-32.
 */
namespace profiles {

namespace ansi {

/**
 * \brief Computes bit masks using a lookup table for the given string.
 *
 * The number of entries in the lookup table is 256 * length of the string. The bit mask for
 * (unsigned) character `c` at string index `i` is stored in entry number `i * 256 + c`.
 */
template <typename T = uint8_t>
class precomputed {
	std::string word;
	std::vector<T> profiles;
public:
	using string_type = std::string;
	using char_type = char;

	precomputed(const string_type& word, int max_distance)
	    : word(word), profiles(word.size() * 256) {
		if (max_distance > 7)
			throw std::runtime_error("maximum supported distance (3) exceeded");
		for (int base = 0; base < length(); base++) {
			int cnt = std::min(length() - base, max_distance * 2 + 1);
			for (int c = 0; c < 256; c++) {
				T row = 0;
				for (int index = cnt-1; index >= 0; --index) {
					row <<= 1;
					if (((uint8_t)word[base + index]) == c)
						row |= 1;
				}
				profiles[base * 256 + c] = row;
			}
		}
	}

	char_type operator[](int pos) const {
		return word[pos] & 0xFF;
	}

	T operator()(int pos, char_type c) const {
		return pos >= length() ? 0 : profiles[pos * 256 + (uint8_t)c];
	}

	int length() const { return word.size(); }
};

#ifdef __SSE4_2__
/**
 * \brief Computes bit masks using the _mm_cmpestrm function.
 *
 * This functor supports distances of up to 3.
 */
class sse {
	int max_str_length;
	std::vector<char> w;

public:
	using string_type = std::string;
	using char_type = char;

	sse(const string_type& word, int max_distance)
	    : max_str_length(max_distance * 2 + 1), w(word.size() + 8) {
		if (max_distance > 3)
			throw std::runtime_error("maximum supported distance (3) exceeded");
		std::copy(word.begin(), word.end(), w.begin());
	}

	int operator()(int pos, char_type c) const {
		__m128i sw = _mm_set_epi64x(0ull, *reinterpret_cast<const uint64_t*>(&w[pos]));
		__m128i sc = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c);
		return _mm_extract_epi32(_mm_cmpestrm(sc, 1, sw, max_str_length /* length*/, _SIDD_LEAST_SIGNIFICANT), 0);
	}

	char_type operator[](int pos) const {
		return w[pos] & 0xFF;
	}

	int length() const { return w.size() - 8; }
};
#endif
} // namespace ansi

namespace utf32 {

/**
 * \brief Computes bit masks by iterating over the input string.
 */
class naive {
	int max_substr_length;
	std::vector<uint32_t> codepoints;

public:
	using char_type = uint32_t;
	using string_type = std::vector<uint32_t>;

	naive(string_type codepoints, int max_distance)
	    : max_substr_length(max_distance * 2 + 1), codepoints(std::move(codepoints)) {}

	int operator()(int pos, char_type c) const {
		int e = std::min(pos + max_substr_length, length());
		uint32_t result = 0;
		for (int i = e - 1; i >= pos; --i) {
			result <<= 1;
			if (codepoints[i] == c)
				result |= 1;
		}
		return result;
	}

	char_type operator[](int pos) const {
		return codepoints[pos];
	}

	int length() const { return codepoints.size(); }
};

} // namespace utf32

namespace utf8 {

/**
 * \brief Converts the input string to UTF-32 and uses profiles::utf32::naive.
 */
class naive : public profiles::utf32::naive {
private:

	static std::vector<uint32_t> convert(const std::string& word) {
		std::vector<uint32_t> result;
		::utf8::to_utf32(word.begin(), word.end(), std::back_inserter(result));
		return result;
	}

public:
	using string_type = std::string;

	naive(const string_type& word, int max_distance)
	    : profiles::utf32::naive(convert(word), max_distance) {}
};

} // namespace utf8

} // namespace profiles
/**
 * \brief Implementation of generic levenshtein automata.
 *
 * For a description of the algorithm, see Schulz, K. U., & Mihov, S. (2002). Fast string
 * correction with Levenshtein automata. International Journal on Document Analysis and Recognition,
 * 5(1), 67-85.
 */
namespace detail {

/**
 * \brief A position in the input word.
 */
struct position {
	/** 
	 * \brief position in the input word
	 */
	int i;
	/**
	 * \brief edit distance so far
	 */
	int e;
	/**
	 * \brief true iff we're matching a transposition
	 */
	bool trans;

	position(int i, int e, bool trans = false) : i(i), e(e), trans(trans) {}

	bool subsumes(position other, int max_distance) const {
		if (other.e <= e)
			return false;

		if (trans) {
			return other.i == i && (other.trans || max_distance == other.e);
		} else {
			auto distance = std::abs(other.i + other.trans - i);
			return distance <= other.e - e;
		}
	}

	bool operator==(position other) const {
		return i == other.i && e == other.e && trans == other.trans;
	}

	/**
	 * \brief Compare two positions.
	 *
	 * this ordering is consistent with subsumes: `a.subsumes(b) -> a < b`
	 *
	 * \param other The position to compare this one with
	 */
	bool operator<(position other) const {
	    return (e < other.e) || (e == other.e && i < other.i) ||
	           (e == other.e && i == other.i && trans < other.trans);
	}

	/**
	 * \brief Compute the positions that are reachable from this position.
	 *
	 * \param max_edit_distance The maximum allowed edit distance
	 * \param word_length The length of the input word
	 * \param allow_trans Whether transpositions are allowed
	 * \param profile The string bit mask, relative to position 0
	 * \param result the set to add all reachable positions to
	 */
	void delta(int max_edit_distance, int word_length, bool allow_trans, uint32_t profile,
	           std::set<position>& result) const {
	    auto add = [=, &result](int i, int e, bool trans = false) {
	        // only add valid positions
	        if (e <= max_edit_distance && i <= word_length) {
	            result.emplace(i, e, trans);
	        }
	    };

	    if (trans) {
	        if ((profile >> i) & 1)
	            add(i + 2, e);
	    } else if (i < word_length && ((profile >> i) & 1)) {
	        add(i + 1, e);
	    } else {
	        add(i, e + 1);
	        add(i + 1, e + 1);

	        int l = std::min(max_edit_distance - e + 1, word_length - i);
	        // find the distance to the first match
	        // -1 -> no match
	        int j = __builtin_ffs(profile >> i) - 1;
	        if (j > 0 && j < l) {
	            add(i + j + 1, e + j);
	            if (allow_trans && j == 1) {
	                add(i, e + 1, true);
	            }
	        }
	    }
	}
};

/**
 * \brief A state of the levenshtein NFA.
 */
struct levenshtein_nfa_state {

	/**
	 * \brief The set of valid positions.
	 */
	std::set<position> positions;

	/**
	 * \brief Offset this state's positions by the given amount.
	 *
	 * This adds `d` to all position's `i` field.
	 *
	 * \param d The delta
	 */
	void offset(int d) {
		std::set<position> new_positions;
		for (auto p : positions)
			new_positions.emplace(p.i + d, p.e, p.trans);
		std::swap(positions, new_positions);
	}

	/**
	 * \brief Get the greatest lower bound of all positions in the input word.
	 */
	int base() const {
		if (positions.empty())
			return 0;
		return std::min_element(positions.begin(), positions.end(),
		                        [](position a, position b) { return a.i < b.i; })->i;
	}

	/**
	 * \brief Compute a state transition
	 *
	 * \param max_distance The automaton's maximum allowed levenshtein distance
	 * \param word_length The length of the matcher's input word
	 * \param trans Whether transpositions are allowed.
	 * \param profile The profile for the input character at the current input word position.
	 * \return The destination state. This may be empty if the input is invalid for the current
	 *         state.
	 */
	levenshtein_nfa_state delta(int max_distance, int word_length, bool trans,
	                            uint32_t profile) const {
		levenshtein_nfa_state result;
		for (auto pos : positions) {
			if (pos.i > word_length)
				//		continue;
				return {}; // position (state) invalid for this word length
			pos.delta(max_distance, word_length, trans, profile, result.positions);
		}
		// remove subsumed
		for (auto it = result.positions.begin(), e = result.positions.end(); it != e;) {
			if (std::any_of(result.positions.begin(), e,
			                [=](const position& p) { return p.subsumes(*it, max_distance); })) {
				it = result.positions.erase(it);
			} else {
				++it;
			}
		}
		return result;
	}

	bool operator==(const levenshtein_nfa_state& other) const {
		return other.positions == positions;
	}

	/**
	 * \brief Get the maximum number of remaining characters for a word to be accepted.
	 *
	 * \param max_distance the automaton's maximum allowed levenshtein distance
	 */
	int max_remaining(int max_distance) const {
		if (positions.empty())
			return -1;
		// the remaining edit distance for a position is max_distance - position.e.
		// Since base() is always 0 when this gets called, we get additional distance for
		// positions where i > 0.
		auto max =
		    *std::max_element(positions.begin(), positions.end(),
		                      [](position max, position x) { return max.i - max.e < x.i - x.e; });
		return max_distance + max.i - max.e;
	}
};

/**
 * \brief A state of the parametric levenshtein DFA
 */
struct levenshtein_automaton_state {
	/**
	 * \brief The state ID
	 */
	int id;

	/**
	 * \brief In a levenshtein_automaton table, this is the position delta.
	 * Otherwise, it's the current position in the input word.
	 */
	int pos;

	levenshtein_automaton_state(int id, int pos) : id(id), pos(pos) {}
	bool valid() const { return id != 0; }
};

/**
 * \brief Class for running the levenshtein NFA and converting it to a DFA.
 */
struct levenshtein_nfa_converter {

	/**
	 * \brief Maximum edit distance of this automaton.
	 */
	int max_distance;

	/**
	 * |brief Whether transpositions are allowed.
	 */
	bool allow_trans;

	/**
	 * \brief (Normalized) states that have already been encountered.
	 *
	 * The states are normalized so that at least one position has `i = 0`.
	 */
	std::vector<levenshtein_nfa_state> states;

	/**
	 * \brief Construct a NFA -> DFA converter.
	 *
	 * \param max_distance the maximum levenshtein distance
	 * \param allow_trans whether to allow transpositions
	 */
	levenshtein_nfa_converter(int max_distance, bool allow_trans)
	    : max_distance(max_distance), allow_trans(allow_trans), states(max_distance + 2) {
		for (int e = 0; e <= max_distance; ++e) {
			states[e + 1].positions.emplace(0, e);
		}
	}

	/**
	 * \brief Get the current number of characters.
	 */
	int size() const { return states.size(); }

	/**
	 * \brief Compute a state transition. This may create a new state, so size() may increase by one
	 * after calling this.
	 *
	 * \param state the index of the current state
	 * \param word_length the length of the input word
	 * \param profile a bit mask. See the profiles namespace.
	 */
	levenshtein_automaton_state next(int state, int word_length, uint32_t profile) {
		levenshtein_nfa_state s =
		    states[state].delta(max_distance, word_length, allow_trans, profile);

		// normalize the state
		int base = s.base();
		s.offset(-base);
		auto it = std::find(states.begin(), states.end(), s);
		const int i = it - states.begin();

		if (it == states.end()) {
			states.emplace_back(std::move(s));
		}
		return { i, base };
	}
};

} // namespace detail

/**
 * \brief A generic levenshtein DFA.
 *
 * This class is used together with levenshtein_matcher.
 */
class levenshtein_automaton {
	/**
	 * \brief Transition tables.
	 *
	 * There is a table for each number of remaining characters in the range [0; max_distance*2+1].
	 */
	std::vector<std::vector<detail::levenshtein_automaton_state>> tables;

	/**
	 * \brief state attributes
	 *
	 * [2*state.id]   -> profile starting with 1 required
	 * [2*state.id+1] -> end of string required
	 */
	std::vector<std::vector<bool>> attributes;

	/**
	 * \brief The maximum value of `word_length - state.pos` for state to be accepting.
	 */
	std::vector<int> max_remaining;

	/**
	 * \brief maximum edit distance of this automaton.
	 */
	int max_distance;

	/**
	 * \brief Check whether a state requires a profile with a set LSB.
	 *
	 * \param remaining the number of remaining characters
	 * \param id the state's id
	 * \return true if all transitions for profiles with no set LSB lead to the invalid state.
	 */
	bool state_requires_profile_1(int remaining, int id) const { return attributes[remaining][id * 2]; }

	/**
	 * \brief Check whether no more characters can be accepted.
	 *
	 * \param remaining the number of remaining characters
	 * \param id the state's id
	 * \return true if all transitions lead to the invalid state.
	 */
	bool state_requires_end_of_string(int remaining, int id) const { return attributes[remaining][id * 2 + 1]; }

	template <typename>
	friend struct levenshtein_matcher;

public:
#ifdef DEBUG
	detail::levenshtein_nfa_converter nfa;
#endif

	/**
	 * \brief Construct a new generic levenshtein automaton.
	 *
	 * \param max_distance the maximum allowed edit distance
	 * \param trans whether to allow transpositions.
	 */
	levenshtein_automaton(int max_distance, bool trans)
	    : tables(2 * max_distance + 2), attributes(2 * max_distance + 2),
	      max_distance(max_distance)
#ifdef DEBUG
	      ,
	      nfa(max_distance, trans) {
#else
	{
		detail::levenshtein_nfa_converter nfa(max_distance, trans);
#endif
		// table index = number of remaining characters
		for (int remaining = 2 * max_distance + 1; remaining >= 0; --remaining) {
			// nfa.size() is not constant!
			for (int state = 0; state < nfa.size(); ++state) {
				bool requires_profile_1 = true;
				bool requires_end_of_string = true;
				// possible profiles for this table
				for (uint32_t profile = 0; profile < (1u << remaining); ++profile) {
					auto next = nfa.next(state, remaining, profile);
					if (!(profile & 1) && (next.id != -1))
						requires_profile_1 = false;
					requires_end_of_string &= (next.id != 1);
					tables[remaining].push_back(next);
				}

				attributes[remaining].push_back(requires_profile_1);
				attributes[remaining].push_back(requires_end_of_string);
			}
		}

		std::transform(
		    nfa.states.begin(), nfa.states.end(), std::back_inserter(max_remaining),
		    [=](const detail::levenshtein_nfa_state& s) { return s.max_remaining(max_distance); });
	}

	detail::levenshtein_automaton_state initial_state() const {
		// id = 1 -> current edit distance 0, position 0 (see levenshtein_nfa_converter constructor)
		// position in input word = 0
		return { 1, 0 };
	};

	/**
	 * \brief Check whether in the given state, an exact match is required.
	 *
	 * \param s the state
	 * \param word_length the length of the input word
	 */
	bool requires_exact_match(detail::levenshtein_automaton_state s, int word_length) const {
		int remaining = std::min(max_distance * 2 + 1, word_length - s.pos);
		return state_requires_profile_1(remaining, s.id);
	}

	/**
	 * \brief Check whether in the given state, no more characters may follow.
	 *
	 * \param s the state
	 * \param word_length the length of the input word
	 */
	bool requires_end_of_string(detail::levenshtein_automaton_state s, int word_length) const {
		int remaining = std::min(max_distance * 2 + 1, word_length - s.pos);
		return state_requires_end_of_string(remaining, s.id);
	}

	/**
	 * \brief Check whether the given state is an accepting state.
	 *
	 * \param s the sate
	 * \param word_length the length of the input word
	 */
	bool accepted(detail::levenshtein_automaton_state s, int word_length) const {
		return max_remaining[s.id] >= word_length - s.pos;
	}

	detail::levenshtein_automaton_state delta(detail::levenshtein_automaton_state state,
	                                          int word_length, int char_dist) const {
		int remaining = std::min(max_distance * 2 + 1, word_length - state.pos);
		auto d = tables[remaining][state.id * (1u << remaining) + char_dist];
		return { d.id, state.pos + d.pos };
	}
};

/**
 * \brief Matcher for a particular automaton and word.
 *
 * This class specialized the generic levenshtein_automaton DFA for a particular word.
 *
 * \tparam P the string profiler to use. This also determines the string encoding. See namespace
 *           profiles for details.
 */
template <typename P = profiles::utf8::naive>
struct levenshtein_matcher {
public:
	const P profiler;
	const levenshtein_automaton& automaton;

	detail::levenshtein_automaton_state state;

	/**
	 * \brief Construct a new levenshtein matcher.
	 *
	 * \param automaton the generic automaton to use
	 * \param word the word to match against
	 */
	levenshtein_matcher(const levenshtein_automaton& automaton, const typename P::string_type& word)
	    : profiler(word, automaton.max_distance), automaton(automaton),
	      state(automaton.initial_state()) {}

	/**
	 * \brief Reset the matcher to its initial state.
	 */
	void reset() { state = automaton.initial_state(); }

	/**
	 * \brief Process a character.
	 *
	 * This method can be called when #valid returns false.
	 *
	 * \param c the character to process.
	 * \return true if the new state is valid
	 */
	bool next(typename P::char_type c) {
		state = automaton.delta(state, profiler.length(), profiler(state.pos, c));
		return valid();
	}

	/**
	 * \brief Check if a particular code point is required to stay in a valid state.
	 *
	 * \return If a code point is required, `{ true, code_point }`. If the end of the
	 *         string is required, `{ true, 0 }`. Otherwise, `{ false, _ }`.
	 */
	std::pair<bool, typename P::char_type> required_code_point() const {
		if (automaton.requires_exact_match(state, profiler.length())) {
			return { true, profiler[state.pos] };
		} else { 
			return { automaton.requires_end_of_string(state, profiler.length()), 0 };
		}
	}

	/**
	 * \brief Check whether it's still possible to get to an accepting state.
	 */
	bool valid() const { return state.valid(); }

	/**
	 * \brief Check whether the matcher currently is in an accepting state.
	 */
	bool accepted() const { return automaton.accepted(state, profiler.length()); }
};

using utf8_levenshtein_matcher = levenshtein_matcher<profiles::utf8::naive>;

} // namespace levenshtein

} // namespace vlad

#endif
