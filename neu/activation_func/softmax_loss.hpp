#ifndef NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
#define NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
//20150528
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/range/traits.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/activation_func/derivative.hpp>
#include <neu/activation_func/derivative_for_loss.hpp>
#include <neu/kernel.hpp>
namespace neu {
	const char softmax_loss_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void softmax_loss(
			__global float* input, const int input_offset,
			__global float* output, const int output_offset,
			const int input_dim)
		{
			const int b = get_global_id(0);
			float m = input[0+b*input_dim+input_offset];
			for(int i = 0; i < input_dim; ++i) {
				if(m < input[i+b*input_dim+input_offset]) {
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
				output[i+b*input_dim+output_offset] =
					exp(input[i+b*input_dim+input_offset]-log_sum);
			}
		}
	);
	
	class softmax_loss {
	public:
		softmax_loss(std::size_t input_dim, std::size_t batch_size,
				boost::compute::context const& context) :
			input_dim_(input_dim), batch_size_(batch_size),
			softmax_loss_kernel_(
				make_kernel(softmax_loss_kernel_source, "softmax_loss", context)) {}

		template<typename InputRange, typename OutputRange>
		decltype(auto) operator()(InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
			NEU_ASSERT(neu::range::distance(output) == neu::range::distance(input));
			enqueue_nd_range_kernel<1>(queue, softmax_loss_kernel_,
				{0}, {batch_size_},
				neu::range::get_buffer(input),
				static_cast<cl_int>(neu::range::get_begin_index(input)),
				neu::range::get_buffer(output),
				static_cast<cl_int>(neu::range::get_begin_index(output)),
				static_cast<cl_int>(input_dim_));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
			NEU_ASSERT_FOR_HEAVY_CALCULATION( //TODO
				boost::compute::all_of(neu::range::begin(output), neu::range::end(output),
				0.f <= boost::compute::lambda::_1, queue));
			NEU_ASSERT_FOR_HEAVY_CALCULATION( //TODO
				boost::compute::all_of(neu::range::begin(output), neu::range::end(output),
				boost::compute::lambda::_1 <= 1.f, queue));
		}

	private:
		std::size_t input_dim_;
		std::size_t batch_size_;
		kernel softmax_loss_kernel_;
	};
	template<>
	class derivative<softmax_loss> : public derivative_for_loss {};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
