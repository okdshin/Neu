#ifndef NEU_ACTIVATION_FUNC_TANH_HPP
#define NEU_ACTIVATION_FUNC_TANH_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/gpu_buffer_range.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, tanh_kernel, (float x), {
		return tanh(x);
	});
	BOOST_COMPUTE_FUNCTION(float, derivative_tanh_kernel, (float x), {
		const float t = tan(x);
		return 1.-t*t;
	});
	class tanh {
	public:
		tanh(std::size_t, std::size_t) {} //TODO
		decltype(auto) operator()(gpu_vector_range input, gpu_vector_range output) {
			NEU_ASSERT(size(output) == size(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			boost::compute::transform(input.begin(), input.end(),
				output.begin(), neu::tanh_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
	template<>
	class derivative<tanh> {
	public:
		derivative(std::size_t, std::size_t) {} //TODO
		decltype(auto) operator()(gpu_vector_range input, gpu_vector_range output) {
			NEU_ASSERT(size(output) == size(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			boost::compute::transform(input.begin(), input.end(),
				output.begin(), neu::derivative_tanh_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_TANH_HPP
