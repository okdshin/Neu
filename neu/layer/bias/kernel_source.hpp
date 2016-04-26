#ifndef NEU_BIAS_LAYER_KERNEL_SOURCE_HPP
#define NEU_BIAS_LAYER_KERNEL_SOURCE_HPP
//20151023
#include <boost/compute/utility/source.hpp>
namespace neu {
	constexpr char bias_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void forward(
			const __global float* input, const int input_offset,
			__global float* output, const int output_offset,
			const __global float* bias,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			const int index = o+input_dim*b;
			output[index+output_offset] = input[index+input_offset] + bias[o];
		}

		__kernel void update(
			const __global float* delta,
			__global float* del_bias,
			const int input_dim, const int batch_size)
		{
			const int o = get_global_id(0);

			float bias_sum = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				bias_sum += delta[o+input_dim*b];
			}
			del_bias[o] = bias_sum/batch_size;
		}
	);
}// namespace neu

#endif //NEU_BIAS_LAYER_KERNEL_SOURCE_HPP
