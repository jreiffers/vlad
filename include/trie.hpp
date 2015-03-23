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
#ifndef VLAD_TRIE_HPP__
#define VLAD_TRIE_HPP__

#include <algorithm>
#include <cstring>
#include <stack>
#include <type_traits>
#include <vector>
#ifdef DEBUG
#include <stdexcept>
#endif

#include "levenshtein.hpp"
#include "chunky_vector.hpp"

namespace vlad {

/**
 * \brief Trie-based data structures.
 *
 * In this namespace, there is a word -> [document ID] dictionary (word_index) and a word counter.
 * Both use unsigned 32 bit integers as document IDs. 
 *
 * Both classes assume that documents are processed in order of their ID.
 */
namespace trie {

/**
 * \brief Used by dictionary for find callbacks.
 */
class vector_wrapper {
private:
	const std::vector<uint32_t>* ptr;
	std::vector<uint32_t> vector;

public:
	vector_wrapper(const std::vector<uint32_t>& vector) : ptr(&vector) {}
	vector_wrapper(uint32_t document) : ptr(nullptr), vector({ document }) {}

	const std::vector<uint32_t>& operator*() const {
		if (ptr)
			return *ptr;
		else
			return vector;
	}

	const std::vector<uint32_t>* operator->() const { return &**this; }
};

/**
 * \brief Values of the word counter index.
 */
struct word_counter_value {
	/**
	 * \brief The number of times this word was encountered.
	 */
	uint32_t count;
	
	/**
	 * \brief When inserting: the ID of the document. The insertion will only be counted if the same
	 * word wasn't inserted before with a document id greater than or equal this value. When
	 * finding: the ID of the last document that contains the word.
	 */
	uint32_t max_document_id;
};

/**
 * \brief Contains the abstract trie implementation.
 *
 * The following illustration shows the internal structure of a vlad trie:
 *
 * \dot
 * digraph structure {
 *     rankdir=LR;
 *     node [shape=record, fontname=Helvetica, fontsize=10];
 *     root [label="{node|{<p0> 0: v|<p1> 1: h|<p2> 2: w}}"]
 *     inode [label="{node|{<p0> 0: e|<p1> 1: o}}"]
 *
 *     leaf1 [label="{{leaf|{l|a|d}}|100}"];
 *     leaf2 [label="{{leaf|{l|l|o}}|142}"];
 *     leaf3 [label="{{leaf|{u|s|e}}|11}"];
 *     leaf4 [label="{{leaf|{o|r|l|d}}|142}"];
 *
 *     root:p0 -> leaf1;
 *     root:p1 -> inode;
 *     root:p2 -> leaf4;
 *     inode:p0 -> leaf2;
 *     inode:p1 -> leaf3;
 * }
 * \enddot
 *
 * This trie represents the map `{ vlad -> 100, hello -> 142, world -> 142, house -> 11 }`. Values
 * are stored in leaves. Common prefixes of keys are stored in inner nodes, unique suffixes up to a
 * certain length in the leaves themselves. The portions of the keys that are stored in the leaves
 * are called "tails".
 *
 * ### Key encoding
 *
 * Keys are assumed to be unicode strings. Unicode code points are encoded as a sequence of
 * `uint16_t` code units as follows:
 * - Code points `c` < 32767 are represented by a single code unit: `c + 1`
 * - Code points `c` >= 32767 are represented by two code units: `[(c>>14)|0x8000, (c&0x3fff)+1]`
 *
 * This encoding ensures that every string is mapped to a sequence of non-zero code units. This is
 * required because the length of the tail is not stored in a leaf, so the zero code unit is used
 * as a marker for end of string.
 *
 * ### Child pointers
 *
 * For improved cache locality, the first code unit of a child node is stored in the parent node,
 * together with the pointer. Pointers are 4 bytes, of which 2 bits are used to keep track of the
 * type of the child node. Therefore, vlad tries support at most ~1 billion different keys. In
 * practice, this number is much lower, because leaves and inner nodes currently use the same
 * address space. However, the code does not assume that this is the case, so that could be changed
 * if necessary.
 *
 * ### Node types
 *
 * All node types have the same size (in bytes). This enables the code to only use a vector to
 * store the children if there are more than a few (currently four). The first four children
 * are stored directly in the inner node. When a fifth child gets added, the node is converted
 * to a node whose children are stored in a vector.
 *
 * Each trie type has its own leaf type. See counter_leaf for an example.
 *
 * Currently, the children of the root node are always stored in a vector.
 *
 * ### Thread safety
 *
 * Currently, it is only safe to concurrently read the tries.
 */
namespace detail {

/**
 * \brief Used as a packed and tagged pointer in tries.
 *
 * This is used to store child node's ID and their type.
 */
struct trie_ptr {
private:
	/**
	 * \brief pointer and tag
	 *
	 * The two most-significant bits are used for the tag, the rest is used for the pointer.
	 */
	uint32_t data;

public:
	/**
	 * \brief The number of bits usable as a pointer.
	 */
	static constexpr int POINTER_BITS = 30;
	/**
	 * \brief Bitmask for the pointer bits.
	 */
	static constexpr uint32_t POINTER_MASK = (1u << POINTER_BITS) - 1;
	/**
	 * \brief The flag used to identify an inner node (as opposed to a leaf)
	 */
	static constexpr int INNER_NODE_FLAG = 0b01;
	/**
	 * \brief The flag used to identify a node_dynamic. Only valid when #INNER_NODE_FLAG = 1.
	 */
	static constexpr int DYNAMIC_NODE_FLAG = 0b10;

	/**
	 * \brief Initialize ptr and flags to zero.
	 */
	trie_ptr() : data(0) {}

	/**
	 * \brief Initialize with the given pointer and flags.
	 *
	 * \param pointer The pointer value. Must be <= #POINTER_BITS.
	 * \param flags The flags. only the two least significant bits are used.
	 */
	trie_ptr(uint32_t pointer, uint32_t flags) : data(pointer | (flags << 30)) {}

	/**
	 * \brief Get the pointer's flags
	 */
	uint32_t flags() const { return data >> 30; }

	/**
	 * \brief Add the given flag(s) using bitwise or.
	 *
	 * \param flags The flags to add. Only the two least significant bits are used.
	 */
	void add_flag(uint32_t flags) { data |= (flags << 30); }

	/**
	 * \brief Get the pointer value.
	 */
	uint32_t ptr() const { return data & 0x3fff'ffffu; }

	/**
	 * \brief Converts to true if the pointer value or the flags are non-zero.
	 */
	operator bool() const { return data; }

	/**
	 * \brief Dereference this pointer.
	 *
	 * Get the node this pointer points to from the given node allocator and call `op` with it.
	 *
	 * \param alloc A node allocator.
	 * \param op something callable with trie_node_fixed, trie_node_dynamic, and
	 *          trie_leaf
	 */
	template <typename Alloc, typename L>
	inline void dereference(Alloc& alloc, L op);

	/**
	 * \brief Dereference this pointer, which does not point at a leaf.
	 *
	 * Get the node this pointer points to from the given node allocator and call `op` with it. This
	 * method may only be called if the pointer doesn't point at a leaf.
	 *
	 * \param alloc A node allocator.
	 * \param op something callable with trie_node_fixed and trie_node_dynamic
	 */
	template <typename Alloc, typename L>
	inline void dereference_no_leaf(Alloc& alloc, L op);

	/**
	 * \brief Dereference this pointer, which points at a leaf.
	 *
	 * Get the leaf this pointer points at from the given node allocator and call `op` with it. This
	 * method may only be called if the pointer points at a leaf.
	 *
	 * \param alloc A node allocator.
	 * \param op something callable with trie_leaf
	 */
	template <typename Alloc, typename L>
	inline void dereference_leaf(Alloc& alloc, L op);

} __attribute__((packed));

/**
 * \brief Helper for converting UTF-32 to the trie encoding.
 */
template <typename I>
struct u16_iter {
	I utf32iter;
	uint32_t u16len;
	bool half_emitted;

	u16_iter(I begin, I end) : utf32iter(begin), u16len(0), half_emitted(false) {
		while (begin != end) {
			if (*begin >= 32767)
				++u16len;
			++u16len;

			++begin;
		}
	}

	uint16_t next() {
		uint16_t result;
		if (u16len == 0) {
			result = 0;
		} else {
			--u16len;
			if (half_emitted) {
				result = (*utf32iter & 0x3fff) + 1;
				half_emitted = false;
				++utf32iter;
			} else if (*utf32iter >= 32767) {
				result = (*utf32iter >> 14) | 0x8000;
				half_emitted = true;
			} else {
				result = *utf32iter + 1;
				++utf32iter;
			}
		}
		return result;
	}
};

/**
 * \brief Base class for trie leaves.
 *
 * A leaf consists of a tail and a value. The tail contains zero or more code units.
 *
 * \tparam T the derived class
 * \tparam V the value type
 */
template <typename T, typename V>
struct trie_leaf {

	using value_type = V;

	static_assert(std::is_trivial<value_type>(), "value type must be trivial");
	static_assert(sizeof(value_type) <= 20, "sizeof(value type) must be <= 20 bytes");

	/**
	 * \brief The maximum number of code units in the tail.
	 */
	static constexpr size_t MAX_TAIL_LENGTH = (24 - sizeof(value_type)) / 2;

	/**
	 * \brief The flags that identify a trie_leaf in a trie_ptr.
	 */
	static constexpr uint32_t NODE_FLAGS = 0;

	/**
	 * \brief The value stored in the leaf.
	 */
	value_type value;

	/**
	 * \brief The tail
	 */
	uint16_t tail[MAX_TAIL_LENGTH];

	/**
	 * \brief Initialize a leaf.
	 *
	 * \param u16iter a code unit iterator
	 * \param u16len the number of code units in `u16iter`. Must be <= MAX_TAIL_LENGTH
	 * \param val the initial value value
	 */
	template <typename Iter>
	trie_leaf(u16_iter<Iter>& iter, const value_type& val);

	/**
	 * \brief Insert a new entry into this leaf or update this leaf.
	 *
	 * This either updates the leaf's value or splits the leaf. If the leaf is split, it is replaced
	 * by a new trie_node_fixed with two children, one of which is this leaf.
	 *
	 * \param alloc A node allocator.
	 * \param self_ptr a reference to the pointer pointing at this leaf
	 * \param iter iterator for the key to insert
	 * \param val the value to insert
	 */
	template <typename Iter, typename Alloc>
	void insert(Alloc& alloc, trie_ptr& self_ptr, u16_iter<Iter>& iter, const value_type& val);

	/**
	 * \brief Check whether this leaf's tail matches the given levenshtein_matcher.
	 *
	 * \param matcher The levenshtein matcher to match against. This method changes the state of the
	 *                matcher.
	 * \param half_code_point if a unicode code point requires two code units (and was therefore
	 *                        split across two layers of the trie), and the first item in the tail
	 *                        contains the second of these code units, this contains the payload of
	 *                        the first code unit.
	 *                        Otherwise, zero.
	 */
	template <typename I>
	bool matches(levenshtein::levenshtein_matcher<I>& matcher, uint32_t half_code_point) const;

	/**
	 * \brief Check whether this leaf's tail matches the given levenshtein_matcher, and call
	 * `callback` if it does.
	 *
	 * \param alloc A node allocator.
	 * \param matcher The levenshtein matcher to match against. This method changes the state of the
	 *                matcher.
	 * \param callback The callback to call. The required signature depends on the leaf type.
	 * \param half_code_point if a unicode code point requires two code units (and was therefore
	 *                        split across two layers of the trie), and the first item in the tail
	 *                        contains the second of these code units, this contains the payload of
	 *                        the first code unit.
	 *                        Otherwise, zero.
	 */
	template <typename Alloc, typename I, typename C>
	void find(const Alloc& alloc, levenshtein::levenshtein_matcher<I>& matcher, C callback,
	          uint32_t half_code_point) const {
		if (matches(matcher, half_code_point)) {
			const T* t = reinterpret_cast<const T*>(this);
			t->call_callback(alloc, callback);
		}
	}

	/**
	 * \brief Split a leaf.
	 *
	 * The leaf's tail is split at the given index.
	 *
	 * For example, splitting the following leaf at `split_index`=1 with `u16iter`=ouse:
	 * \dot
	 * digraph before {
	 *     rankdir=LR;
	 *     node [shape=record, fontname=Helvetica, fontsize=10];
	 *     parent [label="..."]
	 *     leaf [label="leaf|{h|e|l|l|o}"]
	 *     parent -> leaf;
	 * }
	 * \enddot
	 *
	 * Gives the following result:
	 * \dot
	 * digraph after {
	 *     rankdir=LR;
	 *     node [shape=record, fontname=Helvetica, fontsize=10];
	 *     parent [label="..."]
	 *     nodef1 [label="{node fixed|{<p0> 0: h|1: none|2: none|3: none}}"]
	 *     nodef2 [label="{node fixed|{<p0> 0: e|<p1> 1: o|2: none|3: none}}"]
	 *
	 *     leaf1 [label="leaf|{l|l|o}"];
	 *     leaf2 [label="leaf|{u|s|e}"];
	 *
	 *     parent -> nodef1;
	 *     nodef1:p0 -> nodef2;
	 *     nodef2:p0 -> leaf1;
	 *     nodef2:p1 -> leaf2;
	 * }
	 * \enddot
	 *
	 * \param alloc A node allocator
	 * \param self_ptr a reference to the pointer pointing at this leaf
	 * \param split_index the index in #tail to split the leaf at
	 * \param u16iter A code unit iterator
	 * \param u16len The number of code units in `u16iter`
	 * \param val The value of the new leaf
	 */
	template <typename Iter, typename Alloc>
	void split(Alloc& alloc, trie_ptr& self_ptr, size_t split_index, uint16_t head,
	           u16_iter<Iter>& iter, const value_type& val);

	template <typename Alloc, typename C>
	void each(const Alloc& alloc, std::vector<uint32_t>& buffer, C callback,
	          uint32_t code_unit) const;

#ifdef DEBUG
	template <typename Alloc>
	void print(const Alloc& alloc, size_t depth) const;
#endif
} __attribute__((aligned(8)));

/**
 * \brief A leaf whose value is a list of document IDs.
 *
 * This leaf stores either a single document ID or the ID of a vector of document IDs in its value
 * field. Inserting multiple values with the same key adds them to the list.
 */
struct index_leaf : public trie_leaf<index_leaf, uint32_t> {
	using trie_leaf<index_leaf, uint32_t>::trie_leaf;

	static constexpr uint32_t DOCLIST_FLAG = 1u << 31;
	static constexpr uint32_t DOCLIST_MASK = DOCLIST_FLAG - 1;

	/**
	 * \brief Update the leaf's value by adding the new value to the list of documents.
	 *
	 * If there's already a document list in this leaf, add the value to the list. Otherwise, make
	 * the value a list and add both the old value and the new value to the list.
	 *
	 * \param alloc A node allocator
	 * \param val The value to add to this leaf
	 */
	template <typename Alloc>
	void update_value(Alloc& alloc, const value_type& val) {
		if (value & DOCLIST_FLAG) {
			alloc.doclist(value & DOCLIST_MASK).emplace_back(val);
		} else {
			uint32_t list_id;
			std::vector<uint32_t>* list;
			std::tie(list_id, list) = alloc.make_doclist();

			list->emplace_back(value);
			list->emplace_back(val);
			value = DOCLIST_FLAG | list_id;
		}
	}

	/**
	 * \brief Call a callback function with this leaf's value.
	 *
	 * The callback is called with a vector_wrapper.
	 *
	 * \param alloc A node allocator
	 * \param callback The callback to call
	 */
	template <typename Alloc, typename C>
	void call_callback(const Alloc& alloc, C callback) const {
		if (value & DOCLIST_FLAG) {
			callback(vector_wrapper(alloc.doclist(value & DOCLIST_MASK)));
		} else {
			callback(vector_wrapper(value));
		}
	}
};

/**
 * \brief A leaf whose value is a counter.
 *
 * This leaf stores a counter value. Inserting multiple values with the same key adds their values.
 */
struct counter_leaf : public trie_leaf<counter_leaf, word_counter_value> {
	using trie_leaf<counter_leaf, word_counter_value>::trie_leaf;

	/**
	 * \brief Update the leaf's value by summing the old value and the new value.
	 *
	 * \param alloc A node allocator.
	 * \param The value to add to this leaf's value
	 */
	template <typename Alloc>
	void update_value(Alloc&, const word_counter_value& val) {
		if (value.max_document_id < val.max_document_id) {
			value.count += val.count;
			value.max_document_id = val.max_document_id;
		}
	}

	/**
	 * \brief Call a callback function with this leaf's value.
	 *
	 * The callback is called with a `uint32_t`.
	 *
	 * \param alloc A node allocator
	 * \param callback The callback to call
	 */
	template <typename Alloc, typename C, typename... Args>
	void call_callback(const Alloc&, C callback, Args&&... args) const {
		callback(std::forward<Args>(args)..., value);
	}
};

/**
 * \brief Structure used to identify a child in an inner trie node.
 */
struct trie_child {
	/**
	 * \brief The type and id of the child node.
	 */
	trie_ptr next;
	/**
	 * \brief The code unit that leads to this child node.
	 */
	uint16_t code_unit;

	/**
	 * \brief Intialize a trie_child with a null pointer and the given code unit.
	 *
	 * \param codeunit the code unit that leads to the new child node.
	 */
	trie_child(uint16_t code_unit) : next(), code_unit(code_unit) {}

	/**
	 * \brief trie_child instances are ordered by code unit.
	 *
	 * \param other the instance to compare to
	 */
	bool operator<(trie_child other) const { return code_unit < other.code_unit; }
} __attribute__((packed));

/**
 * \brief Super class of inner trie nodes.
 *
 * Implements insert and find. Subclasses need to implement `append_child` and provide
 * `STATIC_SIZE`, `SORTED` and `NODE_FLAGS` constants. See trie_node_fixed for details.
 */
template <typename T>
struct trie_node {

	/**
	 * \brief Insert a new entry into this subtree.
	 *
	 * \param alloc A node allocator.
	 * \param self_ptr a reference to the pointer pointing at this node
	 * \param iter iterator for the key to insert
	 * \param val the value to insert
	 */
	template <typename Iter, typename Alloc>
	void insert(Alloc& alloc, trie_ptr& self_ptr, u16_iter<Iter>& iter,
	            const typename Alloc::value_type& val);

	/**
	 * \brief Find values whose keys match the given levenshtein_matcher.
	 *
	 * \param alloc A node allocator.
	 * \param matcher The levenshtein matcher to match against. This method changes the state of the
	 *                matcher.
	 * \param callback something callable with a std::vector<uint32_t>
	 * \param half_code_point if a unicode code point requires two code units (and was therefore
	 *                        split across two layers of the trie), and this layer is the second of
	 *                        these layers, this contains the payload of the first code unit.
	 *                        Otherwise, zero.
	 */
	template <typename Alloc, typename I, typename C>
	void find(const Alloc& alloc, levenshtein::levenshtein_matcher<I>& matcher, C callback,
	          uint32_t half_code_point) const;

	template <typename Alloc, typename C>
	void each(const Alloc& alloc, std::vector<uint32_t>& buffer, C callback,
	          uint32_t code_unit) const;

#ifdef DEBUG
	template <typename Alloc>
	void print(const Alloc& alloc, size_t depth) const;
#endif
};

/**
 * \brief a trie_node with up to four children.
 */
struct trie_node_fixed : public trie_node<trie_node_fixed> {
	/**
	 * \brief Tells trie_node that the number of children in a trie_node_fixed is constant.
	 */
	static constexpr bool STATIC_SIZE = true;
	/**
	 * \brief Tells trie_node that the children in a trie_node_fixed are not sorted.
	 */
	static constexpr bool SORTED = false;
	/**
	 * \brief The flags of a trie_ptr pointing at a trie_node_fixed.
	 */
	static constexpr uint32_t NODE_FLAGS = trie_ptr::INNER_NODE_FLAG;

	/**
	 * \brief Up to four children. Child pointers that are not yet used are set to zero.
	 */
	trie_child children[4];

	/**
	 * \brief Append a child to this trie_node_fixed by converting it to a trie_node_dynamic.
	 *
	 * After calling this on an object, that object must no longer be used as a trie_node_fixed.
	 *
	 * \param self_ptr The pointer pointing at this node. This will be updated to reflect the new
	 *                 type of this node.
	 * \param alloc A node allocator.
	 * \param code_unit the first code unit of the new child.
	 */
	template <typename Alloc>
	trie_child& append_child(trie_ptr& self_ptr, Alloc& alloc, uint16_t code_unit);
} __attribute__((aligned(8)));

/**
 * \brief a trie_node with any number of children.
 */
struct trie_node_dynamic : public trie_node<trie_node_dynamic> {
	/**
	 * \brief Tells trie_node that the number of children in a trie_node_fixed is not constant.
	 */
	static constexpr bool STATIC_SIZE = false;
	/**
	 * \brief The flags of a trie_ptr pointing at a trie_node_dynamic.
	 */
	static constexpr uint32_t NODE_FLAGS = trie_ptr::INNER_NODE_FLAG | trie_ptr::DYNAMIC_NODE_FLAG;

	/**
	 * \brief The child nodes. Children are sorted by code unit.
	 */
	std::vector<trie_child> children;

	/**
	 * \brief Initialize a node with no children.
	 */
	trie_node_dynamic() {}

	/**
	 * \brief Initialize a node with a range of children.
	 *
	 * The values in the range must already be sorted.
	 *
	 * \param b beginning of a range
	 * \param e end of a range
	 */
	template <typename Iter>
	trie_node_dynamic(Iter b, Iter e)
	    : children(b, e) {}

	/**
	 * \brief Append a child to this node.
	 *
	 * The first two arguments are ignored.
	 *
	 * \param code_unit the first code unit of the child.
	 */
	template <typename Alloc>
	trie_child& append_child(trie_ptr&, Alloc&, uint16_t code_unit) {
		auto it = std::lower_bound(children.begin(), children.end(), trie_child(code_unit));
		it = children.insert(it, trie_child(code_unit));
		return *it;
	}
} __attribute__((aligned(8)));

template <typename Alloc, typename L>
inline void trie_ptr::dereference(Alloc& alloc, L l) {
	switch (flags()) {
	case 0b00:
		l(alloc.template node<typename Alloc::leaf_type>(ptr()));
		break;
	case 0b01:
		l(alloc.template node<trie_node_fixed>(ptr()));
		break;
	case 0b11:
		l(alloc.template node<trie_node_dynamic>(ptr()));
		break;
#ifdef DEBUG
	case 0b10:
		throw std::logic_error("invalid pointer tag");
#endif
	}
}

template <typename Alloc, typename L>
inline void trie_ptr::dereference_no_leaf(Alloc& alloc, L l) {
	switch (flags()) {
	case 0b01:
		l(alloc.template node<trie_node_fixed>(ptr()));
		break;
	case 0b11:
		l(alloc.template node<trie_node_dynamic>(ptr()));
		break;
#ifdef DEBUG
	default:
		throw std::logic_error("invalid pointer tag for dereference_no_leaf");
#endif
	}
}

template <typename Alloc, typename L>
inline void trie_ptr::dereference_leaf(Alloc& alloc, L l) {
#ifdef DEBUG
	switch (flags()) {
	case 0b00:
#endif
		l(alloc.template node<typename Alloc::leaf_type>(ptr()));
#ifdef DEBUG
		break;
	default:
		throw std::logic_error("invalid pointer tag for dereference_leaf");
	}
#endif
}

template <typename T, typename V>
template <typename I>
bool trie_leaf<T, V>::matches(levenshtein::levenshtein_matcher<I>& matcher,
                              uint32_t half_code_point) const {
	// reassemble code points, feed code points to matcher
	uint32_t hp = half_code_point;
	for (size_t i = 0; i < MAX_TAIL_LENGTH && tail[i]; ++i) {
		if (!hp && (tail[i] & 0x8000)) {
			hp = tail[i] & 0x7fff;
		} else {
			uint32_t code_point = (hp << 14) | (tail[i] - 1);
			if (!matcher.next(code_point)) {
				// invalid state
				return false;
			}
			hp = 0;
		}
	}

	return matcher.accepted();
}

/**
 * \brief Create a new leaf.
 *
 * Insert a new leaf at the given position. The leaf is of type `Alloc::leaf_type`.
 *
 * \param alloc A node allocator.
 * \param where The position to create the new leaf at. If `u16len` is > MAX_TAIL_LENGTH, one
 *              or more trie_node_fixed instances will be inserted before the leaf.
 * \param iter iterator for the key
 * \param val The new leaf's initial value
 */
template <typename Iter, typename Alloc>
void insert_new_leaf(Alloc& alloc, trie_ptr& where, u16_iter<Iter> iter,
                     const typename Alloc::value_type& val) {
	trie_ptr* next_ptr = &where;
	while (iter.u16len > Alloc::leaf_type::MAX_TAIL_LENGTH) {
		trie_node_fixed* node;
		std::tie(*next_ptr, node) = alloc.template make_node<trie_node_fixed>();
		node->children[0].code_unit = iter.next();
		next_ptr = &node->children[0].next;
	}

	typename Alloc::leaf_type* leaf;
	std::tie(*next_ptr, leaf) = alloc.template make_node<typename Alloc::leaf_type>();
	new (leaf) typename Alloc::leaf_type(iter, val);
};

#ifdef DEBUG
template <typename T>
template <typename Alloc>
void trie_node<T>::print(const Alloc& alloc, size_t depth) const {
	const T* t = (const T*)this;

	std::cout << "node" << std::endl;
	for (auto child : t->children) {
		if (child.next) {
			for (size_t i = 0; i <= depth; ++i)
				std::cout << " ";
			std::cout << child.code_unit << " -> ";
			child.next.dereference(alloc,
			                       [=, &alloc](const auto& c) { c.print(alloc, depth + 1); });
		}
	}
}

template <typename T, typename V>
template <typename Alloc>
void trie_leaf<T, V>::print(const Alloc&, size_t) const {
	std::cout << "leaf [";

	for (size_t i = 0; i < MAX_TAIL_LENGTH; ++i) {
		if (i)
			std::cout << " ";
		std::cout << tail[i];
	}

	std::cout << "]" << std::endl;
}
#endif

template <typename Alloc>
trie_child& trie_node_fixed::append_child(trie_ptr& self_ptr, Alloc& alloc, uint16_t code_unit) {
	// convert to node_dynamic
	trie_child ch[4] = { children[0], children[1], children[2], children[3] };
	std::sort(std::begin(ch), std::end(ch));
	// after the following line, children[] is garbage!
	auto converted = new (this) trie_node_dynamic(ch, ch + 4);
	alloc.notify_node_dynamic(self_ptr.ptr());
	self_ptr.add_flag(trie_ptr::DYNAMIC_NODE_FLAG);

	return converted->append_child(self_ptr, alloc, code_unit);
}

template <typename T>
template <typename Iter, typename Alloc>
void trie_node<T>::insert(Alloc& alloc, trie_ptr& self_ptr, u16_iter<Iter>& iter,
                          const typename Alloc::value_type& val) {
	uint16_t code_unit = iter.next();
	T* t = (T*)this;

	if (!T::STATIC_SIZE) {
		// t->children is a vector
		auto pos =
		    std::lower_bound(std::begin(t->children), std::end(t->children), trie_child(code_unit));
		if (pos != std::end(t->children) && pos->code_unit == code_unit) {
			pos->next.dereference(alloc, [&](auto& n) { n.insert(alloc, pos->next, iter, val); });
			return;
		}
	} else {
		// t->children is an array
		trie_child* insertion_point = nullptr;
		for (auto child = std::begin(t->children), e = std::end(t->children); child != e; ++child) {
			if (!child->next) {
				if (insertion_point == nullptr) {
					insertion_point = &*child;
				} else {
					memmove(insertion_point + 1, insertion_point,
					        sizeof(trie_child) * (&*child - insertion_point));
				}

				insertion_point->code_unit = code_unit;
				insert_new_leaf(alloc, insertion_point->next, iter, val);
				return;
			} else if (child->code_unit == code_unit) {
				child->next.dereference(alloc,
				                        [&](auto& n) { n.insert(alloc, child->next, iter, val); });
				return;
			} else if ((child->code_unit > code_unit) && insertion_point == nullptr) {
				insertion_point = &*child;
			}
		}
	}

	trie_child& ptr = t->append_child(self_ptr, alloc, code_unit);
	insert_new_leaf(alloc, ptr.next, iter, val);
}

template <typename T>
template <typename Alloc, typename I, typename C>
void trie_node<T>::find(const Alloc& alloc, levenshtein::levenshtein_matcher<I>& matcher,
                        C callback, uint32_t half_code_point) const {
	T* t = (T*)this;

	auto state = matcher.state;
	auto child = std::begin(t->children);
	auto e = std::end(t->children);

#if !defined(VLAD_NO_REQUIRED_CODE_POINT_SHORTCUT)
	if (!T::STATIC_SIZE) {
		auto req = matcher.required_code_point();
		if (req.first) {
			if (req.second) {
				uint32_t code_point = req.second;
				uint16_t next_code_unit;
				uint16_t next_half_code_point = 0;
				if (half_code_point) {
					if ((code_point >> 14) != half_code_point)
						return;
					next_code_unit = (code_point & 0x3fff) + 1;
				} else {
					if (code_point >= 32767) {
						next_half_code_point = (code_point >> 14);
						next_code_unit = 0x8000 | next_half_code_point;
					} else {
						next_code_unit = code_point + 1;
					}
				}

				auto pos = std::lower_bound(std::begin(t->children), std::end(t->children),
											trie_child(next_code_unit));

				if (pos != std::end(t->children) && pos->code_unit == next_code_unit) {
					if (!next_half_code_point) {
						if (!matcher.next(code_point)) {
#ifdef DEBUG
							throw std::logic_error("matcher.next returned false");
#endif
						}
					}
					pos->next.dereference(alloc, [&](auto& n) {
						n.find(alloc, matcher, callback, next_half_code_point);
					});
				}
			} else {
				// end of string
#ifdef DEBUG
				if (!matcher.accepted()) {
					throw std::logic_error("matcher.accepted returned false");
				}
#endif
				if (half_code_point) {
					return;
				}

				// children can be empty if we're the root node and nothing has been inserted yet
				if (__builtin_expect(std::end(t->children) - std::begin(t->children) == 0, 0)) {
					return;
				}

				if (t->children[0].code_unit == 0) {
					t->children[0].next.dereference(alloc, [&](auto& n) {
						n.find(alloc, matcher, callback, 0);
					});
				}
			}
			return;
		}
	}
#endif /* !defined(VLAD_NO_REQUIRED_CODE_POINT_SHORTCUT) */

	for (; child != e; ++child) {
		if (T::STATIC_SIZE && !child->next) {
			return;
		}

		matcher.state = state;
		if (child->code_unit & 0x8000) {
			uint32_t hp = child->code_unit & 0x7fff;
			child->next.dereference(alloc, [=, &alloc, &matcher](const auto& n) {
				n.find(alloc, matcher, callback, hp);
			});
		} else if ((child->code_unit == 0 && matcher.accepted()) ||
		           matcher.next((half_code_point << 14) | (child->code_unit - 1))) {
			child->next.dereference(alloc, [=, &alloc, &matcher](const auto& n) {
				n.find(alloc, matcher, callback, 0);
			});
		}
	}
}

template <typename T>
template <typename Alloc, typename C>
void trie_node<T>::each(const Alloc& alloc, std::vector<uint32_t>& buffer, C callback,
                        uint32_t code_unit) const {
	T* t = (T*)this;

	auto child = std::begin(t->children);
	auto e = std::end(t->children);

	size_t size = buffer.size();

	for (; child != e; ++child) {
		if (T::STATIC_SIZE && !child->next)
			return;

		if (child->code_unit & 0x8000) {
			uint32_t hp = child->code_unit & 0x7fff;
			child->next.dereference(alloc, [=, &alloc, &buffer](const auto& n) {
				n.each(alloc, buffer, callback, hp);
			});
		} else {
			if (child->code_unit != 0) {
				buffer.push_back((code_unit << 14) | (child->code_unit - 1));
			}

			child->next.dereference(
			    alloc, [=, &alloc, &buffer](const auto& n) { n.each(alloc, buffer, callback, 0); });

			buffer.resize(size);
		}
	}
}

template <typename T, typename V>
template <typename Iter>
trie_leaf<T, V>::trie_leaf(u16_iter<Iter>& iter, const value_type& val)
    : value(val) {
	for (uint32_t tail_pos = 0; iter.u16len > 0; ++tail_pos) {
		tail[tail_pos] = iter.next();
	}
}

template <typename T, typename V>
template <typename Iter, typename Alloc>
void trie_leaf<T, V>::split(Alloc& alloc, trie_ptr& self_ptr, size_t split_index, uint16_t head,
                            u16_iter<Iter>& iter, const value_type& val) {
#ifdef DEBUG
	if (split_index >= MAX_TAIL_LENGTH) {
		throw std::logic_error("illegal split_index");
	}
#endif

	trie_ptr this_node = self_ptr;
	trie_ptr* tail_ptr = &self_ptr;

	// insert node_fixeds
	trie_node_fixed* new_node;
	for (uint32_t j = 0; j <= split_index; ++j) {
		std::tie(*tail_ptr, new_node) = alloc.template make_node<trie_node_fixed>();

		// append new node to list
		new_node->children[0].code_unit = tail[j];
		tail_ptr = &new_node->children[0].next;
	}

	// shift code units
	uint32_t new_tail_length = MAX_TAIL_LENGTH - split_index - 1;
	memmove(&tail[0], &tail[split_index + 1], sizeof(tail[0]) * new_tail_length);
	tail[new_tail_length] = 0;
	*tail_ptr = this_node;

	// insert a leaf for the new key
	new_node->children[1].code_unit = head;
	insert_new_leaf(alloc, new_node->children[1].next, iter, val);
}

template <typename T, typename V>
template <typename Iter, typename Alloc>
void trie_leaf<T, V>::insert(Alloc& alloc, trie_ptr& self_ptr, u16_iter<Iter>& iter,
                             const value_type& val) {
	uint32_t tail_pos = 0;

	while (iter.u16len) {
		uint16_t u16 = iter.next();

		// split if there's a mismatch at
		if ((tail[tail_pos] != u16) || (iter.u16len > 0 && tail_pos == MAX_TAIL_LENGTH - 1)) {
			split(alloc, self_ptr, tail_pos, u16, iter, val);
			return;
		}

		++tail_pos;
	}
	if (tail_pos < MAX_TAIL_LENGTH && tail[tail_pos]) {
		split(alloc, self_ptr, tail_pos, 0, iter, val);
		return;
	}

	T* t = reinterpret_cast<T*>(this);
	t->update_value(alloc, val);
}

template <typename T, typename V>
template <typename Alloc, typename C>
void trie_leaf<T, V>::each(const Alloc& alloc, std::vector<uint32_t>& buffer, C callback,
                           uint32_t code_unit) const {
	uint32_t hp = code_unit;
	for (size_t i = 0; i < MAX_TAIL_LENGTH && tail[i]; ++i) {
		if (!hp && (tail[i] & 0x8000)) {
			hp = tail[i] & 0x7fff;
		} else {
			uint32_t codepoint = (hp << 14) | (tail[i] - 1);
			buffer.push_back(codepoint);
			hp = 0;
		}
	}

	const T* t = reinterpret_cast<const T*>(this);
	t->call_callback(alloc, callback, buffer);
}

static_assert(sizeof(index_leaf) == 24, "sizeof(index_leaf) must be 24");
static_assert(sizeof(trie_node_fixed) == 24, "sizeof(trie_node_fixed) must be 24");
static_assert(sizeof(trie_node_dynamic) == 24, "sizeof(trie_node_dynamic) must be 24");

/**
 * \brief The default node allocator.
 *
 * Nodes are stored in a chunky_vector.
 *
 * \tparam L The leaf type.
 * \tparam S The chunk size. See chunky_vector for details.
 */
template <typename L, size_t S = 1024>
struct default_alloc {
	using leaf_type = L;
	using value_type = typename leaf_type::value_type;

	static constexpr size_t NODE_SIZE = sizeof(leaf_type);

	using node_vector_type = vlad::chunky_vector<char[NODE_SIZE], S>;

	node_vector_type node_data;
	std::vector<uint32_t> node5s;
	detail::trie_node_dynamic* root;
	uint32_t leaf_count;

	default_alloc() : node_data(1ull << 30), node5s(), leaf_count(0) {
		uint32_t dummy;
		std::tie(dummy, root) = make_node<detail::trie_node_dynamic>();
		root = new (root) detail::trie_node_dynamic();
	}

	~default_alloc() {
		root->~trie_node_dynamic();
		for (uint32_t id : node5s) {
			node<detail::trie_node_dynamic>(id).~trie_node_dynamic();
		}
	}

	/**
	 * \brief Create a new node.
	 *
	 * The new node's memory is initialized to zero.
	 *
	 * \tparam T The type of the node.
	 */
	template <typename T>
	std::pair<detail::trie_ptr, T*> make_node() {
		static_assert(sizeof(T) == NODE_SIZE, "illegal node type requested");

		if (T::NODE_FLAGS == 0)
			++leaf_count;

		size_t id;
		typename node_vector_type::value_type* addr;

		std::tie(id, addr) = node_data.push_back({ 0 });
		return { { (uint32_t)id, T::NODE_FLAGS }, reinterpret_cast<T*>(addr) };
	}

	/**
	 * \brief Get a reference to a node.
	 *
	 * \param id The ID of the node to get.
	 */
	template <typename T>
	T& node(uint32_t id) {
		return *reinterpret_cast<T*>(&node_data[id]);
	}

	/**
	 * \brief Get a const reference to a node.
	 *
	 * \param id The ID of the node to get.
	 */
	template <typename T>
	const T& node(uint32_t id) const {
		return *reinterpret_cast<const T*>(&node_data[id]);
	}

	/**
	 * \brief Notify the allocator that the given node has been converted to a trie_node_dynamic.
	 *
	 * This must not be called more than once per node ID.
	 *
	 * \param id The ID of the converted node.
	 */
	void notify_node_dynamic(uint32_t id) { node5s.push_back(id); }
};

/**
 * \brief Node allocator for the word_index trie.
 *
 * This class extends default_alloc with
 */
struct word_index_alloc : public default_alloc<index_leaf> {
	using doclist_type = std::vector<uint32_t>;

	vlad::chunky_vector<doclist_type, 128> doclists;

	word_index_alloc() : doclists(1u << 30) {}

	doclist_type& doclist(uint32_t index) { return doclists[index]; }
	const doclist_type& doclist(uint32_t index) const { return doclists[index]; }

	std::pair<uint32_t, doclist_type*> make_doclist() { return doclists.push_back({}); }
};

/**
 * \brief Helper template for counting the number of children of trie_nodes.
 */
template <typename N>
struct child_counter {
	uint32_t operator()(const N&) { return 0; }
};

template <>
struct child_counter<trie_node_fixed> {

	uint32_t operator()(const trie_node_fixed& n) {
		uint32_t result = 0;
		for (auto it = std::begin(n.children), e = std::end(n.children); it != e; ++it) {
			if (it->next)
				++result;
			else
				return result;
		}
		return result;
	}
};

template <>
struct child_counter<trie_node_dynamic> {

	uint32_t operator()(const trie_node_dynamic& n) {
#ifdef DEBUG
		if (n.children.size() > 0xffff) {
			throw std::logic_error("illegal child count");
		}
#endif
		return n.children.size();
	}
};

/**
 * \brief Base class for tries.
 *
 * This class has default implementations of #insert and #find.
 *
 * \tparam A The allocator type
 */
template <typename A>
class basic_trie {
protected:
	A alloc;

public:
	using value_type = typename A::value_type;

	/**
	 * 
	 */
	class state {
		A* alloc;
		trie_ptr node;
		uint32_t child_index;
		uint32_t child_count_;

		void enter_node(trie_ptr n) {
			node = n;
			node.dereference(
			    *alloc, [this](const auto& c) { child_count_ = child_counter<decltype(c)>{}(c); });
		}

	public:
		/**
		 * \brief Check whether we're currently in a leaf.
		 */
		bool is_leaf() const { return (node.flags() & trie_ptr::INNER_NODE_FLAG) == 0; }

		/**
		 * \brief Get the current leaf's tail.
		 *
		 * This may only be called if #is_leaf returns true.
		 */
		const uint16_t(&tail() const)[A::leaf_type::MAX_TAIL_LENGTH] {
			const uint16_t(*result)[A::leaf_type::MAX_TAIL_LENGTH];
			node.dereference_leaf(*alloc, [&result](const auto& c) { result = &c.tail; });
			return *result;
		}

		/**
		 * \brief Get the current leaf's value.
		 *
		 * This may only be called if #is_leaf returns true.
		 */
		const value_type& value() const {
			const value_type* result;
			node.dereference_leaf(*alloc, [&result](const auto& c) { result = &c.value; });
			return *result;
		}

		/**
		 * \brief Get the number of children of the current node.
		 *
		 * Returns 0 if the current node is a leaf.
		 */
		uint32_t child_count() const { return child_count_; }

		/**
		 * \brief Get the current child node's code unit.
		 */
		uint16_t current_child() const {
			uint16_t result;
			node.dereference_no_leaf(*alloc, [&result, this](const auto& c) {
				result = c.children[child_index].code_unit;
			});
			return result;
		}

		/**
		 * \brief Check whether the current node's children are sorted by code unit.
		 */
		bool children_are_sorted() const {
			return (node.flags() & trie_ptr::DYNAMIC_NODE_FLAG) != 0;
		}

		/**
		 * \brief Advance to the next child node.
		 */
		void next_child() { ++child_index; }

		/**
		 * \brief Check whether there is a current child.
		 */
		bool at_end() const { return child_index == child_count_; }

		/**
		 * \brief Enter the current child node.
		 */
		void enter_child() {
			node.dereference_no_leaf(
			    *alloc, [this](const auto& c) { enter_node(c.children[child_index].next); });
		}
	};

	/**
	 * \brief Allows iterating over the trie.
	 */
	state root_state() const { return { &alloc, 0, 0 }; }

	/**
	 * \brief Insert a new value into the trie.
	 *
	 * The key must be UTF-32 encoded.
	 *
	 * \param begin iterator pointing at the beginning of the key
	 * \param end iterator pointing at the end of the key
	 * \param val the value to insert
	 * \tparam The iterator type
	 */
	template <typename Iter>
	void insert(Iter begin, Iter end, const typename A::value_type& val) {
		detail::trie_ptr dummy;
		u16_iter<Iter> iter{ begin, end };
		alloc.root->insert(alloc, dummy, iter, val);
	}

	/**
	 * \brief Find values whose keys match the given levenshtein_matcher and call callback for each.
	 *
	 * \param matcher The levenshtein matcher to match against. After this method returns, the state
	 *                of the matcher is unchanged.
	 * \param callback the callback
	 */
	template <typename I, typename C>
	void find(levenshtein::levenshtein_matcher<I>& matcher, C callback) const {
		auto state = matcher.state;
		alloc.root->find(alloc, matcher, callback, 0);
		matcher.state = state;
	}

	/**
	 * \brief Call the given callback with every key/value pair in this trie.
	 *
	 * \param callback the callback, called with a std::vector<uint32_t> and a value.
	 */
	template <typename C>
	void each(C callback) const {
		std::vector<uint32_t> buffer;
		alloc.root->each(alloc, buffer, callback, 0);
	}

	/**
	 * \brief Get the number of leaves.
	 */
	uint32_t size() const { return alloc.leaf_count; }

#ifdef DEBUG
	void print() const { alloc.root->print(alloc, 0); }
#endif
};

} // namespace detail

/**
 * \brief A word -> [document id] index.
 *
 * Keeps a vector of document IDs for each input word.
 *
 * It is assumed that document IDs are ascending. For instance, it is invalid to first insert
 * "hello" with document ID 2 and then with document ID 1.
 */
class word_index : public detail::basic_trie<detail::word_index_alloc> {};

/**
 * \brief A word -> document count index.
 *
 * It is assumed that document IDs are ascending. For instance, it is invalid to first insert
 * "hello" with document ID 2 and then with document ID 1.
 */
class word_counter : public detail::basic_trie<detail::default_alloc<detail::counter_leaf>> {};

} // namespace trie

} // namespace vlad

#endif
