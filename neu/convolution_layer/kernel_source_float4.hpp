#ifndef NEU_CONVOLUTION_LAYER_KERNEL_SOURCE_HPP
#define NEU_CONVOLUTION_LAYER_KERNEL_SOURCE_HPP
//20151005
#include <boost/compute/kernel.hpp>
namespace neu {
	const char convolution_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void convolution(
			const int input_width, const int output_width,
			const int filter_width,
			const int input_channel_num, const int output_channel_num,
			const int stride, const int pad,
			const __global float* input, __global float* output,
			const __global float* filter)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int m = 0; m < output_channel_num; ++m) {
				float sum = 0.0;
				for(int k = 0; k < input_channel_num; ++k) {
					for(int fr = 0; fr < filter_width; ++fr) {
						for(int fc = 0; fc < filter_width; ++fc) {
							const int ic = oc*stride+fc-pad;
							const int ir = or*stride+fr-pad;
							if(0 <= ic < input_width && 0 <= ir < input_width) {
								const int input_index = index(ic, ir,
									k, b, input_width, input_channel_num);
								const int filter_index = index(fc, fr,
									k, m, filter_width, input_channel_num);
								sum += input[input_index]*filter[filter_index];
							}
						}
					}
				}
				const int output_index = index(oc, or,
					m, b, output_width, output_channel_num);
				output[output_index] = sum;
			}
		}
	);

	const char convolution_back_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		__kernel void convolution_back(
			const __global int* indices_range_list_for_input,
			const __global int* output_indices_list_for_input,
			const __global int* filter_indices_list_for_input,
			const int input_width, const int output_width,
			const int filter_width,
			const int input_channel_num, const int output_channel_num,
			__global float* input, const __global float* output,
			const __global float* filter)
		{
			const int b = get_global_id(1);
			const int i = get_global_id(0);

			for(int k = 0; k < input_channel_num; ++k) {
				float sum = 0.0;
				for(int m = 0; m < output_channel_num; ++m) {
					const int indices_begin = indices_range_list_for_input[i];
					const int indices_end = indices_range_list_for_input[i+1];
					for(int j = indices_begin; j < indices_end; ++j) {
						const int filter_index = index(filter_indices_list_for_input[j],
							k, m, filter_width, input_channel_num);
						const int output_index = index(output_indices_list_for_input[j],
							m, b, output_width, output_channel_num);
						sum += filter[filter_index]*output[output_index];
					}
				}
				const int input_index = index(i, k, b, input_width, input_channel_num);
				input[input_index] = sum;
			}
		}
	);
	const char update_delta_filters_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		__kernel void update_delta_filters(
			const __global int* indices_range_list_for_filter,
			const __global int* input_indices_list_for_filter,
			const __global int* output_indices_list_for_filter,
			const int input_width, const int output_width,
			const int filter_width, const int batch_size,
			const int input_channel_num, const int output_channel_num,
			const __global float* input, const __global float* output,
			__global float* filter)
		{
			const int m = get_global_id(1);
			const int i = get_global_id(0);

			for(int k = 0; k < input_channel_num; ++k) {
				float sum = 0.0;
				for(int b = 0; b < batch_size; ++b) {
					const int indices_begin = indices_range_list_for_filter[i];
					const int indices_end = indices_range_list_for_filter[i+1];
					for(int j = indices_begin; j < indices_end; ++j) {
						const int input_index = index(input_indices_list_for_filter[j],
							k, b, input_width, input_channel_num);
						const int output_index = index(output_indices_list_for_filter[j],
							m, b, output_width, output_channel_num);
						sum += input[input_index]*output[output_index];
					}
				}
				const int filter_index = index(i, k, m, filter_width, input_channel_num);
				filter[filter_index] = sum;
			}
		}
	);
}// namespace neu

#endif //NEU_CONVOLUTION_LAYER_KERNEL_SOURCE_HPP
