#ifndef NEU_AVERAGE_POOLING_LAYER_KERNEL_SOURCE_HPP
#define NEU_AVERAGE_POOLING_LAYER_KERNEL_SOURCE_HPP
//20151026
#include <boost/compute/kernel.hpp>
namespace neu {
	const char average_pooling_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		/*
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		*/
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }

		__kernel void average_pooling(
			const int input_width, const int output_width,
			const int filter_width,
			const int input_channel_num,
			const int stride, const int pad,
			const __global float* input, const int input_offset,
			__global float* output, const int output_offset,
			const __global float* filter)
		{
			/*
			const int b = get_global_id(2);
			const int k = get_global_id(1);
			const int i = get_global_id(0);

			float sum = 0.0;
			const int indices_begin = indices_range_list_for_output[i];
			const int indices_end = indices_range_list_for_output[i+1];
			for(int j = indices_begin; j < indices_end; ++j) {
				const int filter_index = index(filter_indices_list_for_output[j],
					0, 0, filter_width, input_channel_num);
				const int input_index = index(input_indices_list_for_output[j],
					k, b, input_width, input_channel_num);
				sum += filter[filter_index]*input[input_index+input_offset];
			}
			const int output_index = index(i, k, b, output_width, input_channel_num);
			output[output_index+output_offset] = sum;
			*/
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int m = 0; m < input_channel_num; ++m) {
				float sum = 0.0;
				const int fr_end = min(filter_width, pad-or*stride+input_width);
				for(int fr = max(0, pad-or*stride); fr < fr_end; ++fr) {
					const int fc_end = min(filter_width, pad-oc*stride+input_width);
					for(int fc = max(0, pad-oc*stride); fc < fc_end; ++fc) {
						const int ic = oc*stride+fc-pad;
						const int ir = or*stride+fr-pad;
						const int input_index = index(ic, ir,
							m, b, input_width, input_channel_num);
						const int filter_index = index(fc, fr,
							0, 0, filter_width, input_channel_num);
						sum += input[input_index+input_offset]*filter[filter_index];
					}
				}
				const int output_index = index(oc, or,
					m, b, output_width, input_channel_num);
				output[output_index+output_offset] = sum;
			}
		}
	);
	const char average_pooling_back_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int i, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +i; }
		__kernel void average_pooling_back(
			const __global int* indices_range_list_for_input,
			const __global int* output_indices_list_for_input,
			const __global int* filter_indices_list_for_input,
			const int input_width, const int output_width,
			const int filter_width,
			const int input_channel_num,
			__global float* input, const int input_offset,
			const __global float* output, const int output_offset,
			const __global float* filter)
		{
			const int b = get_global_id(2);
			const int k = get_global_id(1);
			const int i = get_global_id(0);

			float sum = 0.0;
			const int indices_begin = indices_range_list_for_input[i];
			const int indices_end = indices_range_list_for_input[i+1];
			for(int j = indices_begin; j < indices_end; ++j) {
				const int filter_index = index(filter_indices_list_for_input[j],
					0, 0, filter_width, input_channel_num);
				const int output_index = index(output_indices_list_for_input[j],
					k, b, output_width, input_channel_num);
				sum += filter[filter_index]*output[output_index+output_offset];
			}
			const int input_index = index(i, k, b, input_width, input_channel_num);
			input[input_index+input_offset] = sum;
		}
	);
}// namespace neu

#endif //NEU_AVERAGE_POOLING_LAYER_KERNEL_SOURCE_HPP
