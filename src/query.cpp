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
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <stack>
#include <stdexcept>
#include <fstream>
#include <limits>

#define DEBUG

#include <levenshtein.hpp>
#include <trie.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <iterator>
#include <set>
#include <queue>
#include <map>

std::string trim(const std::string& s) {
	auto first = s.find_first_not_of(" \n\t\r");
	if (first == std::string::npos) {
		return std::string();
	} else {
		auto last = s.find_last_not_of(" \n\t\r");
		return s.substr(first, last - first + 1);
	}
}

template <typename E>
struct op {
	typedef E element_type;

	virtual ~op() {}

	/**
	 * \brief Produce the next element. Returns true iff there is another element.
	 */
	virtual bool next() = 0;

	/**
	 * \brief Produce the next element that is >= min.
	 */
	virtual bool next(const E& min) = 0;

	/**
	 * \brief Get the element produced by the last call to next() or next(min).
	 *
	 * If the last call returned false, the return value is undefined.
	 */
	virtual const E& get() = 0;
};

template <typename E>
class vector_scan : public op<E> {
private:
	vlad::trie::vector_wrapper base;
	uint32_t pos = ~0u;

public:
	vector_scan(vlad::trie::vector_wrapper base) : base(std::move(base)) {}

	bool next() override { return ++pos < base->size(); }

	bool next(const E& min) override {
		do {
			if (++pos >= base->size())
				return false;
		} while ((*base)[pos] < min);
		return true;
	}

	const E& get() override { return (*base)[pos]; }
};

template <typename E>
class set_union : public op<E> {
private:
	struct queue_entry {
		size_t operand;
		const E* value;

		bool operator<(const queue_entry& other) const {
			return *other.value < *value;
		}

		queue_entry(size_t op_idx, const E* value) : operand(op_idx), value(value) {}
	};

	using queue_t = std::priority_queue<queue_entry>;

	std::vector<std::unique_ptr<op<E>>> operands;
	queue_t queue;
	const E* val;

public:
	set_union(const set_union& rhs) = delete;
	set_union& operator=(const set_union& rhs) = delete;

	set_union(std::vector<std::unique_ptr<op<E>>>& operands) : operands(std::move(operands)), val(nullptr) {
		size_t index = 0;
		for (auto& operand : this->operands) {
			if (operand->next()) {
				queue.emplace(index, &operand->get());
			}
			++index;
		}
	}

	bool next() override {
		bool found = false;
		while (!queue.empty() && !found) {
			auto min_entry = queue.top();
			queue.pop();

			auto& min_op = operands[min_entry.operand];

			// this is the only line that differs from the
			// version with parameter.
			if ((found = (!val || (*val < *min_entry.value)))) {
				val = min_entry.value;
			}

			if (min_op->next()) {
				min_entry.value = &min_op->get();
				queue.push(min_entry);
			}
		}
		return found;
	}

	bool next(const E& min_val) override {
		if (!val || (*val < min_val)) {
			// If next() returns true, val will be overwritten.
			// Otherwise, the result of get() is explicitly undefined.
			val = &min_val;
		}
		return next();
	}

	const E& get() override { return *val; }
};

template <typename E>
class set_intersection : public op<E> {
	std::vector<std::unique_ptr<op<E>>> operands;
	E val;
	E min_value;
	E max_value;

public:
	set_intersection(const set_intersection& rhs) = delete;
	set_intersection& operator=(const set_intersection& rhs) = delete;

	set_intersection(std::vector<std::unique_ptr<op<E>>> operands, E min_value = std::numeric_limits<E>::min(),
	             E max_value = std::numeric_limits<E>::max())
	    : operands(std::move(operands)), min_value(min_value), max_value(max_value) {}

	bool next() {
		return next(min_value);
	}

	bool next(const E& min_val) {
		E min = max_value;
		E max = min_value;

		for (size_t i = 0; i < operands.size(); ++i) {
			if (!operands[i]->next(min_val)) {
				return false;
			}
			E val = operands[i]->get();
			min = std::min(val, min);
			max = std::max(val, max);
		}

		while (min < max) {
			min = max_value;
			for (size_t i = 0; i < operands.size(); ++i) {
				if (operands[i]->get() < max) {
					if (!operands[i]->next(max))
						return false;
					max = operands[i]->get();
				}

				min = std::min(operands[i]->get(), min);
			}
		}

		val = max;
		return true;
	}

	const E& get() { return val; }
};

void load_file(FILE* input, std::vector<size_t>& file_offsets, vlad::trie::word_index& index) {
	std::vector<uint32_t> codepoints;
	char buffer[16384];
	size_t offset = 0;
	while (fgets(buffer, sizeof(buffer), input)) {
		file_offsets.push_back(offset);
		auto it = std::begin(buffer);
		uint32_t codepoint;
		do {
			codepoint = utf8::next_codepoint(it, std::end(buffer));
			if (codepoint <= 32) {
				if (!codepoints.empty()) {
					index.insert(codepoints.begin(), codepoints.end(), file_offsets.size() - 1);
					codepoints.clear();
				}
			} else {
				codepoints.push_back(codepoint);
			}
		} while (codepoint != 0);
		offset = ftell(input);
	}
}

int main(int argc, const char* argv[]) {
	if (argc < 2) {
		std::cerr << "usage: " << argv[0] << " FILE" << std::endl;
		return 1;
	}

	auto input = fopen(argv[1], "r");
	if (!input) {
		std::cerr << "unable to open " << argv[1] << " for reading" << std::endl;
		return 1;
	}

	std::vector<size_t> offsets;
	vlad::trie::word_index index;
	load_file(input, offsets, index);

	int dist = 0;
	bool trans = false;
	vlad::levenshtein::levenshtein_automaton automaton(dist, trans);

	while (true) {
		std::cout << (char)27 << "[32m"
		          << "> " << (char)27 << "[m" << std::flush;
		std::string line;
		std::getline(std::cin, line);
		if (!std::cin.good()) {
			std::cout << std::endl;
			return 0;
		}
		line = trim(line);
		if (!line.empty()) {
			boost::char_separator<char> sep(" ");
			boost::tokenizer<boost::char_separator<char>> tokens(line, sep);
			auto cmd = *tokens.begin();
			auto arg = tokens.begin();
			++arg;

			if (cmd == "trans") {
				if (arg == tokens.end() || (*arg != "on" && *arg != "off")) {
					std::cout << "usage: trans (on|off)" << std::endl;
				} else {
					trans = *arg == "on";
					automaton = vlad::levenshtein::levenshtein_automaton(dist, trans);
				}
			} else if (cmd == "dist") {
				if (arg == tokens.end()) {
					std::cout << "usage: dist <n>" << std::endl;
				} else {
					dist = boost::lexical_cast<int, std::string>(*arg);
					automaton = vlad::levenshtein::levenshtein_automaton(dist, trans);
				}
			} else if (cmd == "find") {
				bool sat = true;
				std::vector<std::unique_ptr<op<uint32_t>>> words;
				for (; sat && (arg != tokens.end()); ++arg) {
					vlad::levenshtein::levenshtein_matcher<> m(automaton, *arg);
					std::vector<std::unique_ptr<op<uint32_t>>> scans;
					index.find(m, [&](const vlad::trie::vector_wrapper& v) {
						scans.push_back(std::make_unique<vector_scan<uint32_t>>(v));
					});
					if (scans.size() == 0) {
						sat = false;
					} else if (scans.size() == 1) {
						words.push_back(std::move(scans[0]));
					} else {
						words.push_back(std::make_unique<set_union<uint32_t>>(scans));
					}
				}

				if (sat) {
					set_intersection<uint32_t> result(std::move(words));
					while (result.next()) {
						std::cout << result.get() << ": ";
						char buffer[16384];
						fseek(input, offsets[result.get()], SEEK_SET);
						fgets(buffer, sizeof(buffer), input);
						std::cout << buffer << std::endl;
					}
				}
			} else if (cmd == "quit") {
				break;
			} else {
				std::cout << "unknown command: " << cmd << std::endl;
			}
		}
	}

	fclose(input);
}
