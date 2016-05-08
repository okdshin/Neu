#ifndef NEU_LAYER_IMPL_MULTIPLE_MATRIX_MULTIPLY_HPP
#define NEU_LAYER_IMPL_MULTIPLE_MATRIX_MULTIPLY_HPP
//20160228
#include <boost/compute/utility/source.hpp>
#include <neu/range/traits.hpp>
namespace neu {
	namespace layer {
		namespace impl {
			constexpr char multiple_matrix_multiply_kernel_source[] =
			BOOST_COMPUTE_STRINGIZE_SOURCE(
				__kernel void multiple_matrix_multiply_more_works_per_thread2_reg_64(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 64;
					const int works_per_thread = 2;
					const int rts = tile_size/works_per_thread;
					const int lpt = works_per_thread*works_per_thread;

					const int tidn = get_local_id(0);
					const int tidm = get_local_id(1);
					const int bat = get_global_id(2);
					const int offsetn = tile_size*get_group_id(0);
					const int offsetm = tile_size*get_group_id(1);
					__local float a_sub[64][64];
					__local float b_sub[64][64+2];
					float a_reg;
					float b_reg[2];
					float sum[2][2] = {};
					const int tile_num = (k_size-1)/tile_size+1;
					for(int t = 0; t < tile_num; ++t) {
						for(int l = 0; l < lpt; ++l) {
							const int tid = tidm*rts+tidn;
							volatile const int id = l*rts*rts+tid;
							const int col = id%tile_size;
							const int row = id/tile_size;
							a_sub[row][col] =
								offsetm+row < m_size && tile_size*t+col < k_size ?
									a[(offsetm+row)*k_size+tile_size*t+col+a_offset] : 0.f;
							b_sub[row][col] =
								tile_size*t+row < k_size && offsetn+col < n_size ?
									//b[bat*k_size*n_size+(tile_size*t+row)*n_size+offsetn+col+b_offset] : 0.f;
									b[bat*k_size*n_size+tile_size*t+row+(offsetn+col)*k_size+b_offset] : 0.f;
						}

						barrier(CLK_LOCAL_MEM_FENCE);

						for(int i = 0; i < tile_size; ++i) {
							for (int wn = 0; wn < works_per_thread; ++wn) {
								const int col = tidn+wn*rts;
								b_reg[wn] = b_sub[i][col];
							}
							for(int wm = 0; wm < works_per_thread; ++wm) {
								const int row = tidm+wm*rts;
								a_reg = a_sub[row][i];
								for(int wn = 0; wn < works_per_thread; ++wn) {
									sum[wm][wn] += a_reg*b_reg[wn];
								}
							}
						}

						barrier(CLK_LOCAL_MEM_FENCE);
					}
					for(int wm = 0; wm < works_per_thread; ++wm) {
						const int global_row = offsetm+tidm+wm*rts;
						for(int wn = 0; wn < works_per_thread; ++wn) {
							const int global_col = offsetn+tidn+wn*rts;
							if(global_col < n_size && global_row < m_size) {
								c[bat*m_size*n_size+global_row*n_size+global_col+c_offset] = sum[wm][wn];
							}
						}
					}
				}
			);

			//
			// for gtx960, gtx980
			//
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) multiple_matrix_multiply(
					RangeA const& a, RangeB const& b, RangeC& c,
					int batch_size,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
				static auto mm_kernel = neu::make_kernel(
					neu::layer::impl::multiple_matrix_multiply_kernel_source,
					"multiple_matrix_multiply_more_works_per_thread2_reg_64",
					queue.get_context());
				mm_kernel.set_args(
					range::get_buffer(a),
					static_cast<cl_int>(range::get_begin_index(a)),
					range::get_buffer(b),
					static_cast<cl_int>(range::get_begin_index(b)),
					range::get_buffer(c),
					static_cast<cl_int>(range::get_begin_index(c)),
					static_cast<cl_int>(m),
					static_cast<cl_int>(n),
					static_cast<cl_int>(k)
				);
				std::size_t global[3] = {
					static_cast<std::size_t>(((n-1)/64+1)*64),
					static_cast<std::size_t>(((m-1)/64+1)*64),
					static_cast<std::size_t>(batch_size)
				};
				std::size_t local[3] = {
					static_cast<std::size_t>(64/2),
					static_cast<std::size_t>(64/2),
					static_cast<std::size_t>(1)
				};
				queue.enqueue_nd_range_kernel(mm_kernel, 3, nullptr, global, local);
			}
		}
	}
}// namespace neu

#endif //NEU_LAYER_IMPL_MULTIPLE_MATRIX_MULTIPLY_HPP
