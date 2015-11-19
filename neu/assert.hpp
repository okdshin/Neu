#ifndef NEU_ASSERT_HPP
#define NEU_ASSERT_HPP
//20151026
#include <cassert>
/*
#define NEU_ASSERT(expr) assert(expr)
#define NEU_ASSERT_FOR_HEAVY_CALCULATION(expr) NEU_ASSERT(expr)
*/
#ifndef NEU_DISABLE_ASSERTION
#	define NEU_ASSERT(expr) assert(expr)
#else
#	define NEU_ASSERT(expr)
#endif
#ifndef NEU_DISABLE_ASSERT_FOR_HEAVY_CALCULATION
#	define NEU_ASSERT_FOR_HEAVY_CALCULATION(expr) NEU_ASSERT(expr)
#else
#	define NEU_ASSERT_FOR_HEAVY_CALCULATION(expr)
#endif

#endif //NEU_ASSERT_HPP
