#ifndef NEU_ACTIVATION_FUNC_IDENTITY_HPP
#define NEU_ACTIVATION_FUNC_IDENTITY_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/gpu_buffer_range.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	class identity {
	public:
		identity(int, int) {} //TODO
		decltype(auto) operator()(gpu_vector_range input, gpu_vector_range output) {
			NEU_ASSERT(size(output) == size(input));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			boost::compute::copy(input.begin(), input.end(), output.begin());
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
		}
	};
	template<>
	class derivative<identity> {
	public:
		derivative(int, int) {} //TODO
		decltype(auto) operator()(
				neu::gpu_vector_range, neu::gpu_vector_range output) const {
			boost::compute::fill(output.begin(), output.end(), 1.f);
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_IDENTITY_HPP
