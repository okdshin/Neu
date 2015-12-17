#ifndef NEU_INNER_PRODUCT_LAYER_KERNEL_SOURCE_HPP
#define NEU_INNER_PRODUCT_LAYER_KERNEL_SOURCE_HPP
//20151023
#include <boost/compute/utility/source.hpp>
namespace neu {
	constexpr char inner_product_forward_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void forward(
			const __global float* input, const int input_offset,
			__global float* output, const int output_offset,
			const __global float* weight,
			const int input_dim, const int output_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			float sum = 0.f;
			for(int i = 0; i < input_dim; ++i) {
				sum += weight[i+input_dim*o]*input[i+input_dim*b+input_offset];
			}
			output[o+output_dim*b+output_offset] = sum;
		}
	);
	constexpr char inner_product_backward_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void backward(
			__global float* input, const int input_offset,
			const __global float* output, const int output_offset,
			const __global float* weight,
			const int input_dim, const int output_dim)
		{
			const int b = get_global_id(1);
			const int i = get_global_id(0);

			float sum = 0.f;
			for(int o = 0; o < output_dim; ++o) {
				sum += weight[i+input_dim*o]*output[o+output_dim*b+output_offset];
			}
			input[i+input_dim*b+input_offset] = sum;
		}
	);
	constexpr char inner_product_update_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void update(
			const __global float* input, const __global float* delta,
			__global float* del_weight,
			const int input_dim, const int output_dim, const int batch_size)
		{
			const int o = get_global_id(1);
			const int i = get_global_id(0);

			float weight_sum = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				weight_sum += delta[o+output_dim*b]*input[i+input_dim*b];
			}
			del_weight[i+input_dim*o] = weight_sum/batch_size;
		}
	);
}// namespace neu

#endif //NEU_INNER_PRODUCT_LAYER_KERNEL_SOURCE_HPP
