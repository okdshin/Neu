#ifndef NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
#define NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/gpu_buffer_range.hpp>
#include <neu/kernel.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	const char softmax_loss_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void softmax_loss(
			__global float* input, int input_offset,
			__global float* output, int output_offset,
			const int input_dim)
		{
			const int b = get_global_id(0);
			float m = input[0+b*input_dim];
			for(int i = 0; i < input_dim; ++i) {
				if(m < input[i+b*input_dim]) {
					m = input[i+b*input_dim+input_offset];
				}
			}
			for(int i = 0; i < input_dim; ++i) {
				input[i+b*input_dim+input_offset] -= m;
			}

			float sum = 0.f;
			for(int i = 0; i < input_dim; ++i) {
				sum += exp(input[i+b*input_dim+input_offset]);
			}
			const float log_sum = log(sum);
			for(int i = 0; i < input_dim; ++i) {
				output[i+b*input_dim+output_offset] = exp(input[i+b*input_dim]-log_sum);
			}
		}
	);
	
	class softmax_loss {
	public:
		softmax_loss(std::size_t input_dim, std::size_t batch_size) :
			input_dim_(input_dim), batch_size_(batch_size),
			softmax_loss_kernel_(make_kernel(softmax_loss_kernel_source, "softmax_loss")),
			output_(input_dim*batch_size) {}

		decltype(auto) operator()(gpu_vector_range input, gpu_vector_range output) {
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input));
			NEU_ASSERT(size(output) == size(input));
			enqueue_nd_range_kernel<1>(softmax_loss_kernel_,
				{0}, {batch_size_},
				input.begin().get_buffer(), static_cast<int>(input.begin().get_index()),
				output.begin().get_buffer(), static_cast<int>(output.begin().get_index()),
				static_cast<int>(input_dim_));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(
				boost::compute::all_of(output_.begin(), output_.end(),
				0.f <= boost::compute::lambda::_1));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(
				boost::compute::all_of(output_.begin(), output_.end(),
				boost::compute::lambda::_1 <= 1.f));
		}

	private:
		std::size_t input_dim_;
		std::size_t batch_size_;
		kernel softmax_loss_kernel_;
		gpu_vector output_;
	};
	template<>
	class derivative<softmax_loss> {
	public:
		derivative(std::size_t, std::size_t) {} //TODO
		decltype(auto) operator()(
				neu::gpu_vector_range, neu::gpu_vector_range output) const {
			boost::compute::fill(output.begin(), output.end(), 1.f);
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
