#ifndef NEU_CONVOLUTION_OPTIMIZED_LAYER_KERNEL_SOURCE_HPP
#define NEU_CONVOLUTION_OPTIMIZED_LAYER_KERNEL_SOURCE_HPP
//20151005
#include <boost/compute/kernel.hpp>
namespace neu {
	namespace layer { 
		constexpr char convolution_optimized_kernel_source[] =
			BOOST_COMPUTE_STRINGIZE_SOURCE(
			__kernel void forward(
				const __global float* a,
				const __global float* b, const int b_offset,
				__global float* c, const int c_offset,
				const int m_size, const int n_size, const int k_size,
				const int input_width, const int filter_width, const int output_width,
				const int input_channel_num, const int output_channel_num,
				const int stride, const int pad
			){
				const int tile_size = 32;

				const int col = get_local_id(0);
				const int row = get_local_id(1);
				const int global_col = tile_size*get_group_id(0)+col;
				const int global_row = tile_size*get_group_id(1)+row;
				const int bat = get_global_id(2);

				__local float a_sub[32][32];
				__local float b_sub[32][32];
				float sum = 0.f;
				const int tile_num = (k_size-1)/tile_size+1;
				for(int t = 0; t < tile_num; ++t) {
					const int tiled_col = tile_size*t+col;
					const int tiled_row = tile_size*t+row;
					a_sub[row][col] = global_row < m_size && tiled_col < k_size ?
						a[global_row*k_size+tiled_col] : 0.f;

					if(tiled_row < k_size && global_col < n_size) {
						const int k = tiled_row/(filter_width*filter_width);
						const int ff = tiled_row%(filter_width*filter_width);
						const int fr = ff/filter_width;
						const int fc = ff%filter_width;

						const int or = global_col/output_width;
						const int oc = global_col%output_width;

						const int ir = or*stride+fr-pad;
						const int ic = oc*stride+fc-pad;
						if(0 <= ir && ir < input_width &&
							0 <= ic && ic < input_width) {
							const int input_index =
								bat*input_channel_num*input_width*input_width+ // batch
								k*input_width*input_width+ // channel
								ir*input_width+ // row
								ic; // col
							b_sub[row][col] = b[input_index+b_offset];
						}
						else {
							b_sub[row][col] = 0.f;
						}
					}
					else {
						b_sub[row][col] = 0.f;
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					for(int i = 0; i < tile_size; ++i) {
						sum += a_sub[row][i]*b_sub[i][col];
					}

					barrier(CLK_LOCAL_MEM_FENCE);
				}
				if(global_col < n_size && global_row < m_size) {
					c[bat*output_channel_num*output_width*output_width+
						global_row*n_size+global_col+c_offset] = sum;
				}
			}
			__kernel void update(
				const __global float* delta,
				const __global float* input,
				__global float* del_filters,
				const int m_size, const int n_size, const int k_size,
				const int batch_size,
				const int input_width, const int filter_width, const int output_width,
				const int input_channel_num, const int output_channel_num,
				const int stride, const int pad
			){
				const int tile_size = 32;

				const int col = get_local_id(0);
				const int row = get_local_id(1);
				const int global_col = tile_size*get_group_id(0)+col;
				const int global_row = tile_size*get_group_id(1)+row;
				const int bat = get_global_id(2);

				__local float a_sub[32][32];
				__local float b_sub[32][32];
				float sum = 0.f;
				const int tile_num = (k_size-1)/tile_size+1;
				for(int bat = 0; bat < batch_size; ++bat) {
					for(int t = 0; t < tile_num; ++t) {
						const int tiled_col = tile_size*t+col;
						const int tiled_row = tile_size*t+row;
						a_sub[row][col] = global_row < m_size && tiled_col < k_size ?
							delta[bat*output_channel_num*output_width*output_width+
								global_row*k_size+tiled_col] : 0.f;

						if(tiled_row < k_size && global_col < n_size) {
							const int k = global_col/(filter_width*filter_width);
							const int ff = global_col%(filter_width*filter_width);
							const int fr = ff/filter_width;
							const int fc = ff%filter_width;

							const int or = tiled_row/output_width;
							const int oc = tiled_row%output_width;

							if(0 <= (or*stride+fr-pad) && (or*stride+fr-pad) < input_width &&
								0 <= (oc*stride+fc-pad) && (oc*stride+fc-pad) < input_width) {
								const int input_index =
									bat*input_channel_num*input_width*input_width+ // batch
									k*input_width*input_width+ // channel
									(or*stride+fr-pad)*input_width+ // row
									oc*stride+fc-pad; // col
								b_sub[row][col] = input[input_index];
							}
							else {
								b_sub[row][col] = 0.f;
							}
						}
						else {
							b_sub[row][col] = 0.f;
						}

						barrier(CLK_LOCAL_MEM_FENCE);

						for(int i = 0; i < tile_size; ++i) {
							sum += a_sub[row][i]*b_sub[i][col];
						}
					}

					barrier(CLK_LOCAL_MEM_FENCE);
				}
				if(global_col < n_size && global_row < m_size) {
					del_filters[global_row*n_size+global_col] = sum/batch_size;
				}
			}
			__kernel void backward(
				const __global float* delta, const int delta_offset,
				const __global float* filters,
				__global float* prev_delta, const int prev_delta_offset,
				const int m_size, const int n_size, const int k_size,
				const int input_width, const int filter_width, const int output_width,
				const int input_channel_num, const int output_channel_num,
				const int stride, const int pad
			){
				const int tile_size = 32;

				const int col = get_local_id(0);
				const int row = get_local_id(1);
				const int global_col = tile_size*get_group_id(0)+col;
				const int global_row = tile_size*get_group_id(1)+row;
				const int bat = get_global_id(2);

				__local float a_sub[32][32];
				__local float b_sub[32][32];
				float sum = 0.f;
				const int tile_num = (k_size-1)/tile_size+1;
				for(int t = 0; t < tile_num; ++t) {
					const int tiled_col = tile_size*t+col;
					const int tiled_row = tile_size*t+row;
					if(global_row < m_size && tiled_col < k_size) {
						const int m = tiled_col/(filter_width*filter_width);
						const int ff = tiled_col%(filter_width*filter_width);
						const int fr = ff/filter_width;
						const int fc = ff%filter_width;
						const int k = global_row;

						const int filters_index =
							m*input_channel_num*filter_width*filter_width+
							k*filter_width*filter_width+
							fr*filter_width+
							fc;
						a_sub[row][col] = filters[filters_index];
					}
					else {
						a_sub[row][col] = 0.f;
					}

					if(tiled_row < k_size && global_col < n_size) {
						const int m = tiled_row/(filter_width*filter_width);
						const int ff = tiled_row%(filter_width*filter_width);
						const int fr = ff/filter_width;
						const int fc = ff%filter_width;

						const int ir = global_col/input_width;
						const int ic = global_col%input_width;

						if((ir-fr+pad)%stride == 0 && (ic-fc+pad)%stride == 0) {
							const int or = (ir-fr+pad)/stride;
							const int oc = (ic-fc+pad)/stride;
							if(0 <= or && or < output_width &&
								0 <= oc && oc < output_width) {
								const int output_index =
									bat*output_channel_num*output_width*output_width+
									m*output_width*output_width+
									or*output_width+
									oc;
								b_sub[row][col] = delta[output_index+delta_offset];
							}
							else {
								b_sub[row][col] = 0.f;
							}
						}
						else {
							b_sub[row][col] = 0.f;
						}
					}
					else {
						b_sub[row][col] = 0.f;
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					for(int i = 0; i < tile_size; ++i) {
						sum += a_sub[row][i]*b_sub[i][col];
					}

					barrier(CLK_LOCAL_MEM_FENCE);
				}
				if(global_col < n_size && global_row < m_size) {
					prev_delta[bat*input_channel_num*input_width*input_width+
						global_row*n_size+global_col+prev_delta_offset] = sum;
				}
			}
		);
		template<typename InputRange, typename OutputRange>
		decltype(auto) convolution_optimized_forward(
				neu::kernel& ker,
				InputRange const& input, gpu_vector const& filters, OutputRange& output,
				int batch_size,
				int input_width, int filter_width, int output_width,
				int input_channel_num, int output_channel_num,
				int stride, int pad,
				boost::compute::command_queue& queue) {
			const int m = output_channel_num;
			const int n = output_width*output_width;
			const int k = filter_width*filter_width*input_channel_num;
			std::size_t global[3] = {
				static_cast<std::size_t>(((n-1)/32+1)*32),
				static_cast<std::size_t>(((m-1)/32+1)*32),
				static_cast<std::size_t>(batch_size)
			};
			std::size_t local[3] = {
				static_cast<std::size_t>(32),
				static_cast<std::size_t>(32),
				static_cast<std::size_t>(1)
			};
			ker.set_args(
				filters,
				range::get_buffer(input),
				static_cast<cl_int>(range::get_begin_index(input)),
				range::get_buffer(output),
				static_cast<cl_int>(range::get_begin_index(output)),
				static_cast<cl_int>(m),
				static_cast<cl_int>(n),
				static_cast<cl_int>(k),
				static_cast<cl_int>(input_width),
				static_cast<cl_int>(filter_width),
				static_cast<cl_int>(output_width),
				static_cast<cl_int>(input_channel_num),
				static_cast<cl_int>(output_channel_num),
				static_cast<cl_int>(stride),
				static_cast<cl_int>(pad)
			);
			queue.enqueue_nd_range_kernel(ker, 3, nullptr, global, local);
		}
		template<typename InputRange>
		decltype(auto) convolution_optimized_update(
				neu::kernel& ker,
				InputRange const& delta, gpu_vector const& input, gpu_vector& del_filters,
				int batch_size,
				int input_width, int filter_width, int output_width,
				int input_channel_num, int output_channel_num,
				int stride, int pad,
				boost::compute::command_queue& queue) {
			const int m = output_channel_num;
			const int n = filter_width*filter_width*input_channel_num;
			const int k = output_width*output_width;
			std::size_t global[3] = {
				static_cast<std::size_t>(((n-1)/32+1)*32),
				static_cast<std::size_t>(((m-1)/32+1)*32),
				static_cast<std::size_t>(batch_size)
			};
			std::size_t local[3] = {
				static_cast<std::size_t>(32),
				static_cast<std::size_t>(32),
				static_cast<std::size_t>(1)
			};
			ker.set_args(
				delta,
				input,
				del_filters,
				static_cast<cl_int>(m),
				static_cast<cl_int>(n),
				static_cast<cl_int>(k),
				static_cast<cl_int>(batch_size),
				static_cast<cl_int>(input_width),
				static_cast<cl_int>(filter_width),
				static_cast<cl_int>(output_width),
				static_cast<cl_int>(input_channel_num),
				static_cast<cl_int>(output_channel_num),
				static_cast<cl_int>(stride),
				static_cast<cl_int>(pad)
			);
			queue.enqueue_nd_range_kernel(ker, 3, nullptr, global, local);
		}
		template<typename InputRange, typename OutputRange>
		decltype(auto) convolution_optimized_backward(
				neu::kernel& ker,
				InputRange const& delta, gpu_vector const& filters, OutputRange& prev_delta,
				int batch_size,
				int input_width, int filter_width, int output_width,
				int input_channel_num, int output_channel_num,
				int stride, int pad,
				boost::compute::command_queue& queue) {
			const int m = input_channel_num;
			const int n = input_width*input_width;
			const int k = filter_width*filter_width*output_channel_num;
			std::size_t global[3] = {
				static_cast<std::size_t>(((n-1)/32+1)*32),
				static_cast<std::size_t>(((m-1)/32+1)*32),
				static_cast<std::size_t>(batch_size)
			};
			std::size_t local[3] = {
				static_cast<std::size_t>(32),
				static_cast<std::size_t>(32),
				static_cast<std::size_t>(1)
			};
			ker.set_args(
				range::get_buffer(delta),
				static_cast<cl_int>(range::get_begin_index(delta)),
				filters,
				range::get_buffer(prev_delta),
				static_cast<cl_int>(range::get_begin_index(prev_delta)),
				static_cast<cl_int>(m),
				static_cast<cl_int>(n),
				static_cast<cl_int>(k),
				static_cast<cl_int>(input_width),
				static_cast<cl_int>(filter_width),
				static_cast<cl_int>(output_width),
				static_cast<cl_int>(input_channel_num),
				static_cast<cl_int>(output_channel_num),
				static_cast<cl_int>(stride),
				static_cast<cl_int>(pad)
			);
			queue.enqueue_nd_range_kernel(ker, 3, nullptr, global, local);
		}
	}
}// namespace neu

#endif //NEU_CONVOLUTION_OPTIMIZED_LAYER_KERNEL_SOURCE_HPP
