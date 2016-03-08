#ifndef NEU_CONVOLUTION_OPTIMIZED_LAYER_KERNEL_SOURCE_HPP
#define NEU_CONVOLUTION_OPTIMIZED_LAYER_KERNEL_SOURCE_HPP
//20151005
#include <boost/compute/kernel.hpp>
namespace neu {
	constexpr char convolution_optimized_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void reorder_input(
			const __global float* input, const int input_offset,
			__global float* reordered_input,
			const int input_width,
			const int output_width,
			const int filter_width,
			const int input_channel_num,
			const int output_channel_num,
			const int stride, const int pad)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int k = 0; k < input_channel_num; ++k) {
			for(int fr = 0; fr < filter_width; ++fr) {
			for(int fc = 0; fc < filter_width; ++fc) {
				const int ir = or*stride+fr-pad;
				const int ic = oc*stride+fc-pad;
				const int filter_index =
					k*filter_width*filter_width+
					fr*filter_width+
					fc;
				const int output_index =
					or*output_width+
					oc;
				const int input_index =
					k*input_width*input_width+
					ir*input_width+
					ic;
				const int reordered_input_index =
					b*output_width*output_width
						*filter_width*filter_width*input_channel_num+
					output_index*filter_width*filter_width*input_channel_num+
					filter_index;
				/*
				const int reordered_input_index =
					b*output_width*output_width
						*filter_width*filter_width*input_channel_num+
					filter_index*output_width*output_width+
					output_index;
				*/
				if(0 <= ir && ir < input_width && 0 <= ic && ic < input_width) {
					reordered_input[reordered_input_index] =
						input[input_index+
							b*input_channel_num*input_width*input_width+
							input_offset];
				}
				else {
					reordered_input[reordered_input_index] = 0.f;
				}
			}}}
		}

		__kernel void forward(
			const __global float* reordered_input,
			const __global float* filters,
			__global float* output, const int output_offset,
			const int output_width,
			const int filter_width,
			const int input_channel_num,
			const int output_channel_num)
		{
			const int b = get_global_id(2);
			const int m = get_global_id(1);
			const int o = get_global_id(0);

			float sum = 0.f;
			const int row_size = filter_width*filter_width*input_channel_num;
			const int col_size = output_width*output_width;
			for(int i = 0; i < row_size; ++i) {
				const int filters_index =
					m*row_size+
					i;
				const int reordered_input_index =
					b*row_size*col_size+
					o*row_size+
					i;
				sum += filters[filters_index]*reordered_input[reordered_input_index];
			}
			const int output_index =
				b*col_size*output_channel_num+
				m*col_size+
				o;
			output[output_index+output_offset] = sum;
		}

		__kernel void reorder_filters(
			const __global float* filters,
			__global float* reordered_filters,
			const int filter_width,
			const int input_channel_num,
			const int output_channel_num
		) {
			const int m = get_global_id(2);
			const int k = get_global_id(1);
			const int fcfr = get_global_id(0);
			const int filters_index =
				fcfr+
				k*filter_width*filter_width+
				m*input_channel_num*filter_width*filter_width;
			const int reordered_filters_index =
				fcfr+
				m*filter_width*filter_width+
				k*output_channel_num*filter_width*filter_width;
			reordered_filters[reordered_filters_index] = filters[filters_index];
		}

		__kernel void reorder_delta(
			const __global float* delta, const int delta_offset,
			__global float* reordered_delta,
			const int input_width,
			const int output_width,
			const int filter_width,
			const int input_channel_num,
			const int output_channel_num,
			const int stride, const int pad)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int m = 0; m < output_channel_num; ++m) {
			for(int fr = 0; fr < filter_width; ++fr) {
			for(int fc = 0; fc < filter_width; ++fc) {
				const int ir = or*stride+fr-pad;
				const int ic = oc*stride+fc-pad;
				const int filter_index =
					m*filter_width*filter_width+
					fr*filter_width+
					fc;
				const int output_index =
					m*output_width*output_width+
					or*output_width+
					oc;
				const int input_index =
					ir*input_width+
					ic;
				const int reordered_delta_index =
					b*output_width*output_width
						*filter_width*filter_width*output_channel_num+
					input_index*filter_width*filter_width*output_channel_num+
					filter_index;
				if(0 <= ir && ir < input_width && 0 <= ic && ic < input_width) {
					reordered_delta[reordered_delta_index] =
						delta[output_index+
							b*output_channel_num*output_width*output_width+
							delta_offset];
				}
				else {
					reordered_delta[reordered_delta_index] = 0.f;
				}
			}
			}
			}
		}

		__kernel void backward(
			const __global float* reordered_delta,
			const __global float* reordered_filters,
			__global float* prev_delta, const int prev_delta_offset,
			const int input_width,
			const int filter_width,
			const int input_channel_num,
			const int output_channel_num)
		{
			const int b = get_global_id(2);
			const int k = get_global_id(1);
			const int o = get_global_id(0);

			float sum = 0.f;
			const int row_size = filter_width*filter_width*output_channel_num;
			const int col_size = input_width*input_width;
			for(int i = 0; i < row_size; ++i) {
				const int filters_index =
					k*row_size+
					i;
				const int reordered_delta_index =
					b*row_size*col_size+
					o*row_size+
					i;
				sum += reordered_filters[filters_index]
					*reordered_delta[reordered_delta_index];
			}
			const int prev_delta_index =
				b*col_size*output_channel_num+
				k*col_size+
				o;
			prev_delta[prev_delta_index+prev_delta_offset] = sum;
		}

		__kernel void update(
			const __global float* reordered_input, const __global float* delta,
			__global float* del_filters,
			const int ffk_size,
			const int oo_size,
			const int output_channel_num,
			const int batch_size)
		{
			const int m = get_global_id(1);
			const int ffk= get_global_id(0);

			float weight_sum = 0.f;
			for(int b = 0; b < batch_size; ++b) {
				for(int oo = 0; oo < oo_size; ++oo) {
					const int delta_index =
						b*oo_size*output_channel_num+
						m*oo_size+
						oo;
					const int reordered_index =
						b*oo_size*ffk_size+
						oo*ffk_size+
						ffk;
					weight_sum += delta[delta_index]*reordered_input[reordered_index];
				}
			}
			del_filters[ffk+m*ffk_size] = weight_sum/batch_size;
		}
		/*
		__kernel void update(
			const __global float* reordered_input, const __global float* delta,
			__global float* del_filters,
			const int ffk_size, const int oob_size,
			const int batch_size)
		{
			const int m = get_global_id(1);
			const int ffk= get_global_id(0);

			float weight_sum = 0.f;
			for(int oob = 0; oob < oob_size; ++oob) {
				weight_sum += delta[oob+m*oob_size]*reordered_input[ffk+oob*ffk_size];
			}
			del_filters[ffk+m*ffk_size] = weight_sum/batch_size;
		}
		*/
	);
}// namespace neu

#endif //NEU_CONVOLUTION_OPTIMIZED_LAYER_KERNEL_SOURCE_HPP
