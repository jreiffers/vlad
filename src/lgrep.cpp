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

#include <levenshtein.hpp>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace vlad::levenshtein;

void run(int distance, bool allow_trans, bool whole_word, const string& text) {
	// build a levenshtein automaton for the given maximum edit distance
	levenshtein_automaton a(distance, allow_trans);

	// build a matcher for the search text
	// initializing a matcher isn't completely free, so we reuse one
	levenshtein_matcher<> m(a, text);

	string line;
	while (!cin.eof()) {
		if (getline(cin, line)) {
			// whether the line is already accepted
			bool accepted = false;
			m.reset();

			for (auto it = line.begin(), e = line.end(); it != e && !accepted; ++it) {
				if (*it <= ' ') {
					if (whole_word)
						accepted = m.accepted();
					// restart matching
					m.reset();
				} else {
					// process the character. next returns true if it is still possible to get a
					// match. Here, it doesn't matter, since we need to process the whole line
					// anyway.
					m.next(*it);

					if (!whole_word)
						accepted = m.accepted();
				}
			}

			if (accepted || (whole_word && m.accepted()))
				cout << line << endl;
		}
	}
}

int main(int argc, const char* argv[]) {
	try {
		po::options_description desc("Allowed options");
		desc.add_options()                                                                     //
		    ("help", "display this help text and exit")                                        //
		    ("distance,d", po::value<int>()->default_value(1), "maximum levenshtein distance") //
		    ("whole-word,w", "match whole words (instead of prefixes)")                        //
		    ("transpositions,t", "allow transpositions")                                       //
		    ("text,e", po::value<string>(), "search text");

		po::positional_options_description pd;
		pd.add("text", 1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);
		po::notify(vm);

		if (vm.count("help") || !vm.count("text")) {
			cout << "Usage: " << argv[0] << " [--help] [-d DISTANCE] [-tw] PATTERN" << endl;
			cout << desc << endl;
			return 1;
		}

		int distance = vm["distance"].as<int>();
		bool allow_trans = vm.count("transpositions") > 0;
		bool whole_word = vm.count("whole-word") > 0;
		string text = vm["text"].as<string>();

		run(distance, allow_trans, whole_word, text);
	} catch (po::error& e) {
		cout << e.what() << endl;
		return 1;
	}
}

