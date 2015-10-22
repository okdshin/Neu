#ifndef NEU_AS_CONST_HPP
#define NEU_AS_CONST_HPP
//20151021
#include <type_traits>
namespace neu {
	template<typename T>
	inline typename std::add_const_t<T>&
	as_const( T &t ) noexcept {
		return t;
	}
}// namespace neu

#endif //NEU_AS_CONST_HPP
