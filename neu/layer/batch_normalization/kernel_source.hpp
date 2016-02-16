#ifndef NEU_BATCH_NORMALIZATION_LAYER_KERNEL_SOURCE_HPP
#define NEU_BATCH_NORMALIZATION_LAYER_KERNEL_SOURCE_HPP
//20151023
#include <boost/compute/utility/source.hpp>
namespace neu {
	constexpr char batch_normalization_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void mean(
			const __global float* input, const int input_offset,
			__global float* mean,
			const int batch_size,
			const int input_dim)
		{
			const int o = get_global_id(0);

			float sum = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				sum += input[o+input_dim*b+input_offset];
			}
			mean[o] = sum/batch_size;
		}

		__kernel void variance(
			const __global float* input, const int input_offset,
			__global float* variance,
			const __global float* mean,
			const int batch_size,
			const int input_dim)
		{
			const int o = get_global_id(0);

			float sum = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				const float t = (input[o+input_dim*b+input_offset]-mean[o]);
				sum += t*t;
			}
			variance[o] = sum/batch_size;
		}

		__kernel void normalize_input(
			const __global float* input, const int input_offset,
			__global float* normalized_input,
			const __global float* mean,
			const __global float* variance,
			float eta,
			const int batch_size,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			normalized_input[o+input_dim*b] =
				(input[o+input_dim*b+input_offset]-mean[o])/sqrt(variance[o]+eta);
		}

		__kernel void forward(
			const __global float* normalized_input,
			__global float* output, const int output_offset,
			const __global float* gamma,
			const __global float* beta,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			output[o+input_dim*b+output_offset] =
				gamma[o]*normalized_input[o+input_dim*b]+beta[o];
		}

		__kernel void test_forward(
			const __global float* input, const int input_offset,
			__global float* output, const int output_offset,
			const __global float* gamma,
			const __global float* beta,
			const __global float* total_variance,
			const __global float* total_mean,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			output[o+input_dim*b+output_offset] =
				gamma[o]*rsqrt(total_variance[o]+eta)
					*(input[o+input_dim*b+input_offset]-total_mean[o])
				+beta[o];
		}

		__kernel void del_normalized_input(
			const __global float* delta, const int delta_offset,
			const __global float* gamma,
			__global float* del_normalized_input,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			del_normalized_input[o+input_dim*b] =
				gamma[o]*delta[o+input_dim*b+delta_offset];
		}

		__kernel void del_variance(
			const __global float* input, int input_offset,
			const __global float* del_normalized_input,
			const __global float* mean,
			const __global float* variance,
			float eta,
			__global float* del_variance,
			const int batch_size,
			const int input_dim)
		{
			const int o = get_global_id(0);

			float sum = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				const float dnx = del_normalized_input[o+input_dim*b];
				const float x = input[o+input_dim*b+input_offset];
				sum += dnx*(x-mean[o])*(-1.f/2.f)*powr(variance[o]+eta, -3.f/2.f);
			}
			del_variance[o] = sum;
		}

		__kernel void del_mean(
			const __global float* input, int input_offset,
			const __global float* del_normalized_input,
			const __global float* mean,
			const __global float* variance,
			const __global float* del_variance,
			float eta,
			__global float* del_mean,
			const int batch_size,
			const int input_dim)
		{
			const int o = get_global_id(0);

			float term1 = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				const float dnx = del_normalized_input[o+input_dim*b];
				term1 += dnx*-1.f*rsqrt(variance[o]+eta);
			}

			/*
			float term2 = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				const float x = input[o+input_dim*b+input_offset];
				term2 += -2.f*(x-mean[o]);
			}
			*/

			del_mean[o] = term1;// + del_variance[o]*term2/batch_size;
		}

		__kernel void backward(
			const __global float* input, int input_offset,
			const __global float* del_normalized_input,
			const __global float* mean,
			const __global float* variance,
			const __global float* del_variance,
			const __global float* del_mean,
			float eta,
			__global float* prev_delta, int prev_delta_offset,
			const int batch_size,
			const int input_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);
			
			const float dnx = del_normalized_input[o+input_dim*b];
			const float x = input[o+input_dim*b+input_offset];

			prev_delta[o+input_dim*b+prev_delta_offset] =
				dnx*rsqrt(variance[o]+eta)+
				(del_variance[o]*2.f*(x-mean[o])+del_mean[o])/batch_size;
		}

		__kernel void del_gamma(
			const __global float* normalized_input,
			const __global float* delta,
			__global float* del_gamma,
			int batch_size,
			int input_dim)
		{
			const int o = get_global_id(0);

			float sum = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				const float d = delta[o+input_dim*b];
				const float x = normalized_input[o+input_dim*b];
				sum += d*x;
			}
			del_gamma[o] = sum/batch_size;
		}

		__kernel void del_beta(
			const __global float* delta,
			__global float* del_beta,
			int batch_size,
			int input_dim)
		{
			const int o = get_global_id(0);

			float sum = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				sum += delta[o+input_dim*b];
			}
			del_beta[o] = sum/batch_size;
		}
	);
}// namespace neu

#endif //NEU_BATCH_NORMALIZATION_LAYER_KERNEL_SOURCE_HPP
