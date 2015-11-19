#ifndef NEU_MAX_POOLING_LAYER_KERNEL_SOURCE_HPP
#define NEU_MAX_POOLING_LAYER_KERNEL_SOURCE_HPP
//20151026
#include <boost/compute/kernel.hpp>
namespace neu {
	const char max_pooling_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }

		__kernel void max_pooling(
			const __global float* input, const int input_offset,
			__global float* output, const int output_offset,
			__global int* indices, const int stride, const int input_channel_num,
			const int input_width, const int output_width, const int filter_width)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int k = 0; k < input_channel_num; ++k) {
				float max_val = 0.0;
				int max_index = 0;
				const int r_start = max(0, or*stride-filter_width/2);
				const int r_end =
					min(input_width, or*stride-filter_width/2+filter_width);
				for(int r = r_start; r < r_end; ++r) {
					const int c_start = max(0, oc*stride-filter_width/2);
					const int c_end = 
						min(input_width, oc*stride-filter_width/2+filter_width);
					for(int c = c_start; c < c_end; ++c) {
						const int input_index = input_offset+index(c, r, k, b,
							input_width, input_channel_num);
						const float input_val = input[input_index];
						if(max_val < input_val) {
							max_val = input_val;
							max_index = input_index;
						}
					}
				}
				const int output_index =
					output_offset+index(oc, or, k, b, output_width, input_channel_num);
				output[output_index] = max_val;
				indices[output_index] = max_index;
			}
		}
	);
}// namespace neu

#endif //NEU_MAX_POOLING_LAYER_KERNEL_SOURCE_HPP
