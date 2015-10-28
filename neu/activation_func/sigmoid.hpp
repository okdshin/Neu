#ifndef NEU_ACTIVATION_FUNC_SIGMOID_HPP
#define NEU_ACTIVATION_FUNC_SIGMOID_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/gpu_buffer_range.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, sigmoid_kernel, (float x), {
		return 1./(1.+exp(-x));
	});
	BOOST_COMPUTE_FUNCTION(float, derivative_sigmoid_kernel, (float x), {
		const float sigma = 1./(1.+exp(-x));
		return sigma*(1.-sigma);
	});
	class sigmoid {
	public:
		sigmoid_loss(std::size_t, std::size_t) {} //TODO
		decltype(auto) operator()(gpu_vector_range input, gpu_vector_range output) {
			NEU_ASSERT(size(output) == size(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			boost::compute::transform(input.begin(), input.end(),
				output.begin(), sigmoid_loss_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
	template<>
	class derivative<sigmoid> {
	public:
		derivative(std::size_t, std::size_t) {} //TODO
		decltype(auto) operator()(gpu_vector_range input, gpu_vector_range output) {
			NEU_ASSERT(size(output) == size(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			boost::compute::transform(input.begin(), input.end(),
				output.begin(), neu::derivative_sigmoid_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_SIGMOID_HPP
