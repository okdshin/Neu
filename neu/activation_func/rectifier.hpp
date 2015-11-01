#ifndef NEU_ACTIVATION_FUNC_RECTIFIER_HPP
#define NEU_ACTIVATION_FUNC_RECTIFIER_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/range_traits.hpp>
#include <neu/range_algorithm.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, rectifier_kernel, (float x), {
		return x > 0 ? x : 0;
	});
	BOOST_COMPUTE_FUNCTION(float, derivative_rectifier_kernel, (float x), {
		return x > 0 ? 1 : 0;
	});
	class rectifier {
	public:
		template<typename InputRange, typename OutputRange>
		decltype(auto) operator()(InputRange const& input, OutputRange const& output) {
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			neu::range_transform(input, output, rectifier_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
	template<>
	class derivative<rectifier> {
	public:
		template<typename InputRange, typename OutputRange>
		decltype(auto) operator()(InputRange const& input, OutputRange const& output) {
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			neu::range_transform(input, output, neu::derivative_rectifier_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_RECTIFIER_HPP
