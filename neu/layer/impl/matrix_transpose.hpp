#ifndef NEU_LAYER_IMPL_MATRIX_TRANSPOSE_HPP
#define NEU_LAYER_IMPL_MATRIX_TRANSPOSE_HPP
//20160307

namespace neu {
	namespace layer {
		namespace impl {
			constexpr char matrix_transpose_kernel_source[] =
			BOOST_COMPUTE_STRINGIZE_SOURCE(
				__kernel void matrix_transpose(
					const __global float* input, const int input_offset,
					__global float* output, const int output_offset,
					const int p_size, const int q_size)
				{
					const int tile_size_x = 32;
					const int tile_size_y = 32;

					const int tx = get_local_id(0);
					const int ty = get_local_id(1);
					const int col = get_group_id(0)*tile_size_x+tx;
					const int row = get_group_id(1)*tile_size_y+ty;

					__local float buffer[32][32];

					if(col < q_size && row < p_size) {
						buffer[ty][tx] = input[row*q_size+col+input_offset];
					}
					barrier(CLK_LOCAL_MEM_FENCE);

					const int new_col = get_group_id(1)*tile_size_y+tx;
					const int new_row = get_group_id(0)*tile_size_x+ty;
					if(new_col < p_size && new_row < q_size) {
						output[new_row*p_size+new_col+output_offset] = buffer[tx][ty];
					}
				}
			);

			template<typename InputRange, typename OutputRange>
			decltype(auto) matrix_transpose(
					InputRange const& input, OutputRange& output,
					int row_size, int col_size,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(row_size*col_size == range::distance(input));
				static auto transpose_kernel =
					neu::make_kernel(neu::layer::impl::matrix_transpose_kernel_source,
					"matrix_transpose", queue.get_context());
				transpose_kernel.set_args(
					range::get_buffer(input),
					static_cast<cl_int>(range::get_begin_index(input)),
					range::get_buffer(output),
					static_cast<cl_int>(range::get_begin_index(output)),
					static_cast<cl_int>(row_size),
					static_cast<cl_int>(col_size));
				std::size_t global[2] = {
					static_cast<std::size_t>(((col_size-1)/32+1)*32),
					static_cast<std::size_t>(((row_size-1)/32+1)*32)
				};
				std::size_t local[2] = {
					static_cast<std::size_t>(32),
					static_cast<std::size_t>(32)
				};
				queue.enqueue_nd_range_kernel(transpose_kernel, 2, nullptr, global, local);
			}
		}
	}
}// namespace neu

#endif //NEU_LAYER_IMPL_MATRIX_TRANSPOSE_HPP
