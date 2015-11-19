#ifndef NEU_ACTIVATION_FUNC_TANH_HPP
#define NEU_ACTIVATION_FUNC_TANH_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/range_traits.hpp>
#include <neu/range_algorithm.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, tanh_kernel, (float x), {
		return tanh(x);
	});
	BOOST_COMPUTE_FUNCTION(float, derivative_tanh_kernel, (float x), {
		const float t = tanh(x);
		return 1.f-t*t;
	});
	class tanh {
	public:
		template<typename InputRange, typename OutputRange>
		decltype(auto) operator()(InputRange const& input, OutputRange const& output) {
			NEU_ASSERT(neu::range_distance(output) == neu::range_distance(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			neu::range_transform(input, output, neu::tanh_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
	template<>
	class derivative<tanh> {
	public:
		template<typename InputRange, typename OutputRange>
		decltype(auto) operator()(InputRange const& input, OutputRange const& output) {
			NEU_ASSERT(neu::range_distance(output) == neu::range_distance(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			neu::range_transform(input, output, neu::derivative_tanh_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_TANH_HPP
