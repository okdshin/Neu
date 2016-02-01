#ifndef NEU_LAYER_LOCAL_CONTRAST_NORMALIZATION_KERNEL_SOURCE_HPP
#define NEU_LAYER_LOCAL_CONTRAST_NORMALIZATION_KERNEL_SOURCE_HPP
//20151225

namespace neu {
	namespace layer {
		constexpr char local_mean_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			int index(int x, int y, int c, int id, int width, int channel_num) {
				return id*channel_num*width*width +c*width*width +y*width +x; }

			__kernel void local_mean(
				const int input_width, const int output_width,
				const int filter_width,
				const int input_channel_num,
				const int stride, const int pad,
				const __global float* input, const int input_offset,
				__global float* output, const int output_offset)
			{
				const int b = get_global_id(2);
				const int or = get_global_id(1);
				const int oc = get_global_id(0);

				for(int k = 0; k < input_channel_num; ++k) {
					float sum = 0.0;
					const int fr_end = min(filter_width, pad-or*stride+input_width);
					for(int fr = max(0, pad-or*stride); fr < fr_end; ++fr) {
						const int fc_end = min(filter_width, pad-oc*stride+input_width);
						for(int fc = max(0, pad-oc*stride); fc < fc_end; ++fc) {
							const int ic = oc*stride+fc-pad;
							const int ir = or*stride+fr-pad;
							const int input_index = index(ic, ir,
								k, b, input_width, input_channel_num);

							sum += input[input_index+input_offset];
						}
					}
					const int output_index = index(oc, or,
						k, b, output_width, input_channel_num);
					output[output_index+output_offset] = sum/(filter_width*filter_width);
				}
			}
		);
		constexpr char local_variance_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			int index(int x, int y, int c, int id, int width, int channel_num) {
				return id*channel_num*width*width +c*width*width +y*width +x; }

			__kernel void local_variance(
				const int input_width, const int output_width,
				const int filter_width,
				const int input_channel_num,
				const int stride, const int pad,
				const __global float* input, const int input_offset,
				__global float* output, const int output_offset,
				const __global float* local_mean)
			{
				const int b = get_global_id(2);
				const int or = get_global_id(1);
				const int oc = get_global_id(0);

				for(int k = 0; k < input_channel_num; ++k) {
					float sum = 0.0;
					const int fr_end = min(filter_width, pad-or*stride+input_width);
					for(int fr = max(0, pad-or*stride); fr < fr_end; ++fr) {
						const int fc_end = min(filter_width, pad-oc*stride+input_width);
						for(int fc = max(0, pad-oc*stride); fc < fc_end; ++fc) {
							const int ic = oc*stride+fc-pad;
							const int ir = or*stride+fr-pad;
							const int input_index = index(ic, ir,
								k, b, input_width, input_channel_num);

							sum += pow(input[input_index+input_offset]
									-local_mean[input_index], 2.f);
						}
					}
					const int output_index = index(oc, or,
						k, b, output_width, input_channel_num);
					output[output_index+output_offset] = sum/(filter_width*filter_width);
				}
			}
		);

		constexpr char local_contrast_normalization_forward_kernel_source[] =
			BOOST_COMPUTE_STRINGIZE_SOURCE(
			__kernel void forward(
				const __global float* input, const int input_offset,
				__global float* output, const int output_offset,
				const float alpha, const float beta,
				const __global float* local_mean,
				const __global float* local_variance)
			{
				const int i = get_global_id(0);
				output[i] = input[i]/powr(1.f+alpha*local_variance[i], beta);
			}
		);

		constexpr char local_contrast_normalization_backward_kernel_source[] =
			BOOST_COMPUTE_STRINGIZE_SOURCE(
			__kernel void backward(
				__global float* prev_delta, const int prev_delta_offset,
				const __global float* delta, const int delta_offset,
				const int filter_width,
				const float alpha, const float beta,
				const __global float* local_mean,
				const __global float* local_variance,
				const __global float* input,
				const __global float* output)
			{
				const int i = get_global_id(0);
				const int N = filter_width*filter_width;

				prev_delta[i] = delta[i]*output[i]*(
					(1.f/(/*0.00001f+*/input[i]))
					-(2*alpha*beta*(input[i]+local_mean[i]*(1-N)))
					/(1+(alpha/N)*local_variance[i]));
			}
		);
	}
}// namespace neu

#endif //NEU_LAYER_LOCAL_CONTRAST_NORMALIZATION_KERNEL_SOURCE_HPP
