#ifndef NEU_LAYER_SHARED_DROPOUT_KERNEL_SOURCE_HPP
#define NEU_LAYER_SHARED_DROPOUT_KERNEL_SOURCE_HPP
//20151023
#include <boost/compute/utility/source.hpp>
namespace neu {
	constexpr char shared_dropout_test_forward_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void test_forward(
			const __global float* input, const int input_offset,
			__global float* output, const int output_offset,
			const float probability,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			output[o+input_dim*b+output_offset] = 
				probability*input[o+input_dim*b+input_offset];
		}
	);
	constexpr char shared_dropout_forward_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void forward(
			const __global float* input, const int input_offset,
			__global float* output, const int output_offset,
			const __global float* mask,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			output[o+input_dim*b+output_offset] =
				mask[o]*input[o+input_dim*b+input_offset];
		}
	);
	constexpr char shared_dropout_backward_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void backward(
			__global float* input, const int input_offset,
			const __global float* output, const int output_offset,
			const __global float* mask,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int i = get_global_id(0);

			input[i+input_dim*b+input_offset] =
				mask[i]*output[i+input_dim*b+output_offset];
		}
	);
}// namespace neu

#endif //NEU_LAYER_SHARED_DROPOUT_KERNEL_SOURCE_HPP
