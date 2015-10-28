#ifndef NEU_ACTIVATION_FUNC_RECTIFIER_HPP
#define NEU_ACTIVATION_FUNC_RECTIFIER_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/gpu_buffer_range.hpp>
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
		rectifier(std::size_t, std::size_t) {}
		decltype(auto) operator()(gpu_vector_range input, gpu_vector_range output) {
			NEU_ASSERT(size(output) == size(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			boost::compute::transform(input.begin(), input.end(),
				output.begin(), rectifier_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
	template<>
	class derivative<rectifier> {
	public:
		derivative(std::size_t, std::size_t) {}
		decltype(auto) operator()(gpu_vector_range input, gpu_vector_range output) {
			NEU_ASSERT(size(output) == size(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			boost::compute::transform(input.begin(), input.end(),
				output.begin(), neu::derivative_rectifier_kernel);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	private:
		gpu_vector output_;
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_RECTIFIER_HPP
