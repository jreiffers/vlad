#ifndef VLAD_CHUNKY_VECTOR_HPP__
#define VLAD_CHUNKY_VECTOR_HPP__

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sys/mman.h>
#include <type_traits>
#include <vector>

namespace vlad {
namespace detail {

/**
 * \brief Copy a trivially copyable value.
 *
 * This is used for types that are trivially copyable, but not assignable (e.g. `foo[n]`).
 *
 * \param dest The destination
 * \param src The source
 * \tparam T The type
 */
template <typename T>
std::enable_if_t<std::is_trivially_copyable<T>::value> copy(T* dest, const T& src) {
	memcpy(dest, &src, sizeof(T));
}

/**
 * \brief Copy a non-trivially copyable value.
 *
 * This is used for types that are not trivially copyable, but assignable.
 *
 * \param dest The destination
 * \param src The source
 * \tparam T The type
 */
template <typename T>
std::enable_if_t<!std::is_trivially_copyable<T>::value> copy(T* dest, const T& src) {
	*dest = src;
}

} // namespace detail

/**
 * \brief A stable vector that uses one pointer per `chunk_size` elements.
 *
 * This class allows multiple threads to concurrently insert elements (using #push_back). The
 * maximum size of the vector needs to be known ahead of time. Memory for elements is allocated in
 * chunks of `chunk_size` elements at a atime.
 *
 * \tparam T The element type.
 * \tparam chunk_size The number of elements per chunk.
 */
template <typename T, size_t chunk_size = 1024>
class chunky_vector {
private:
	using array_type = std::array<T, chunk_size>;

	/**
	 * \brief Array of pointers to chunks.
	 */
	array_type** vector;

	/**
	 * \brief The current number of elements in the chunky_vector.
	 *
	 * Some elements may not yet be initialized.
	 */
	size_t n;

	/**
	 * \brief The size of #vector
	 */
	size_t array_size;

public:
	using value_type = T;
	using reference = value_type&;
	using const_reference = const value_type&;
   
	chunky_vector(size_t max_size) : n(0) {
		if (max_size + chunk_size - 1 < max_size) {
			array_size = (max_size / chunk_size) + 1;
		} else {
			array_size = (max_size + chunk_size - 1) / chunk_size;
		}

#ifdef MAP_ANONYMOUS
		auto flags = MAP_PRIVATE | MAP_ANONYMOUS;
#else
		auto flags = MAP_PRIVATE | MAP_ANON;
#endif
		vector = (array_type**)mmap(NULL, array_size * sizeof(array_type*), PROT_READ | PROT_WRITE,
		                            flags, -1, 0);
	}

	chunky_vector(const chunky_vector&) = delete;

	~chunky_vector() {
		for (size_t i = 0; i * chunk_size < n; ++i) {
			delete vector[i];
		}
		munmap(vector, array_size * sizeof(array_type*));
	}

	/**
	 * \brief Add an element to the back of the vector.
	 *
	 * \param val the element to add
	 * \return a pair of the index at which the element was added and a pointer to the element in
	 *         the vector.
	 */
	std::pair<size_t, T*> push_back(const value_type& val) {
		size_t spot = __sync_fetch_and_add(&n, 1);
	try_again:
		array_type** array = &vector[spot / chunk_size];
		if (!*array) {
			array_type* new_array = new array_type();
			if (!__sync_bool_compare_and_swap(array, nullptr, new_array)) {
				delete new_array;
				goto try_again;
			}
		}

		auto addr = &(**array)[spot % chunk_size];
		detail::copy(addr, val);
		return { spot, addr };
	}

	/**
	 * \brief Get the current size of the vector.
	 */
	size_t size() const {
		return n;
	}

	reference operator[](size_t i) {
#ifdef DEBUG
		if (i >= n)
			throw std::runtime_error("index out of bounds");
#endif
		return (*vector[i / chunk_size])[i % chunk_size];
	}

	const_reference operator[](size_t i) const {
#ifdef DEBUG
		if (i >= n)
			throw std::runtime_error("index out of bounds");
#endif
		return (*vector[i / chunk_size])[i % chunk_size];
	}
};

} // namespace vlad

#endif
