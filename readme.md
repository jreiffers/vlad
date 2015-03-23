Vlad is a lightweight header-only library for fuzzy string search. It includes
an implementation of efficient generic levenshtein automata and a generic trie
for indexing documents and counting unique words. 

There are two demo programs. `lgrep` is like a basic grep with fuzzy matching. 
`query` indexes a text file by line and also does fuzzy search.

# Prerequisites

You'll need boost, cmake and a recent clang++. Also, doxygen, if you want 
HTML documentation.

# Getting started

1. Create a `build` directory next to src.
2. Run `cmake ..` in the `build` directory.
3. Run `make all test`.
4. Run `make doc` if you want an HTML version of the documentation.
4. Take a look at `lgrep` and `query`.
