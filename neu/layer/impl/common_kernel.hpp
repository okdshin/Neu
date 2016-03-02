#ifndef NEU_LAYER_IMPL_COMMON_KERNEL_HPP
#define NEU_LAYER_IMPL_COMMON_KERNEL_HPP
//20160228
#include <boost/compute/utility/source.hpp>
#include <neu/range/traits.hpp>
namespace neu {
	namespace layer {
		namespace impl {
			constexpr char common_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
				__kernel void matrix_multiply_normal(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int n = get_global_id(0);
					const int m = get_global_id(1);
					float acc = 0.f;
					for (int k = 0; k < k_size; ++k) {
						acc += a[m*k_size + k] * b[k*n_size + n];
					}
					c[m*n_size + n] = acc;
				}
				__kernel void matrix_multiply_tiled(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 32;

					const int col = get_local_id(0);
					const int row = get_local_id(1);
					const int global_col = tile_size*get_group_id(0)+col;
					const int global_row = tile_size*get_group_id(1)+row;
					__local float a_sub[32][32];
					__local float b_sub[32][32];
					float sum = 0.f;
					const int tile_num =
						(k_size-1)/tile_size+1;
					for(int t = 0; t < tile_num; ++t) {
						const int tiled_col = tile_size*t+col;
						const int tiled_row = tile_size*t+row;
						a_sub[row][col] = global_row < m_size && tiled_col < k_size ?
							a[global_row*k_size+tiled_col+a_offset] : 0.f;
						//b_sub[row][col] = tiled_row < k_size ?
							//b[global_col*k_size+tiled_row+b_offset] : 0.f;
						b_sub[row][col] = tiled_row < k_size && global_col < n_size ?
							b[tiled_row*n_size+global_col+b_offset] : 0.f;

						barrier(CLK_LOCAL_MEM_FENCE);

						for(int i = 0; i < tile_size; ++i) {
							sum += a_sub[row][i]*b_sub[i][col];
						}

						barrier(CLK_LOCAL_MEM_FENCE);
					}
					if(global_col < n_size && global_row < m_size) {
						c[global_row*n_size+global_col+c_offset] = sum;
					}
				}
				__kernel void matrix_multiply_more_works_per_thread2(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 32;
					const int works_per_thread = 2;
					const int rts =
						tile_size/works_per_thread;

					const int col = get_local_id(0);
					const int row = get_local_id(1);
					const int global_col = tile_size*get_group_id(0)+col;
					const int global_row = tile_size*get_group_id(1)+row;
					__local float a_sub[32][32];
					__local float b_sub[32][32];
					float sum[2] = {};
					const int tile_num =
						(k_size-1)/tile_size+1;
					for(int t = 0; t < tile_num; ++t) {
						const int tiled_col = tile_size*t+col;
						const int tiled_row = tile_size*t+row;
						for(int w = 0; w < works_per_thread; ++w) {
							a_sub[row][col+w*rts] =
								global_row < m_size && (tiled_col+w*rts) < k_size ?
									a[global_row*k_size+tiled_col+w*rts+a_offset] : 0.f;
							//b_sub[row][col] = tiled_row < k_size ?
								//b[global_col*k_size+tiled_row+b_offset] : 0.f;
							b_sub[row][col+w*rts] =
								tiled_row < k_size && (global_col+w*rts) < n_size ?
									b[tiled_row*n_size+global_col+w*rts+b_offset] : 0.f;
						}

						barrier(CLK_LOCAL_MEM_FENCE);

						for(int i = 0; i < tile_size; ++i) {
							for(int w = 0; w < works_per_thread; ++w) {
								sum[w] += a_sub[row][i]*b_sub[i][col+w*rts];
							}
						}

						barrier(CLK_LOCAL_MEM_FENCE);
					}
					for(int w = 0; w < works_per_thread; ++w) {
						if(global_col+w*rts < n_size && global_row < m_size) {
							c[global_row*n_size+global_col+w*rts+c_offset] = sum[w];
						}
					}
				}
				__kernel void matrix_multiply_more_works_per_thread4(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 32;
					const int works_per_thread = 4;
					const int rts =
						tile_size/works_per_thread;

					const int col = get_local_id(0);
					const int row = get_local_id(1);
					const int global_col = tile_size*get_group_id(0)+col;
					const int global_row = tile_size*get_group_id(1)+row;
					__local float a_sub[32][32];
					__local float b_sub[32][32];
					float sum[4] = {};
					const int tile_num =
						(k_size-1)/tile_size+1;
					for(int t = 0; t < tile_num; ++t) {
						const int tiled_col = tile_size*t+col;
						const int tiled_row = tile_size*t+row;
						for(int w = 0; w < works_per_thread; ++w) {
							a_sub[row][col+w*rts] =
								global_row < m_size && (tiled_col+w*rts) < k_size ?
									a[global_row*k_size+tiled_col+w*rts+a_offset] : 0.f;
							//b_sub[row][col] = tiled_row < k_size ?
								//b[global_col*k_size+tiled_row+b_offset] : 0.f;
							b_sub[row][col+w*rts] =
								tiled_row < k_size && (global_col+w*rts) < n_size ?
									b[tiled_row*n_size+global_col+w*rts+b_offset] : 0.f;
						}

						barrier(CLK_LOCAL_MEM_FENCE);

						for(int i = 0; i < tile_size; ++i) {
							for(int w = 0; w < works_per_thread; ++w) {
								sum[w] += a_sub[row][i]*b_sub[i][col+w*rts];
							}
						}

						barrier(CLK_LOCAL_MEM_FENCE);
					}
					for(int w = 0; w < works_per_thread; ++w) {
						if(global_col+w*rts < n_size && global_row < m_size) {
							c[global_row*n_size+global_col+w*rts+c_offset] = sum[w];
						}
					}
				}
				__kernel void matrix_multiply_more_works_per_thread2_non_bank_conflict(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 32;
					const int works_per_thread = 2;
					const int rts =
						tile_size/works_per_thread;

					const int col = get_local_id(0);
					const int row = get_local_id(1);
					const int global_col = tile_size*get_group_id(0)+col;
					const int global_row = tile_size*get_group_id(1)+row;
					__local float a_sub[32][32];
					__local float b_sub[32][32+2];
					float sum[2] = {};
					const int tile_num =
						(k_size-1)/tile_size+1;
					for(int t = 0; t < tile_num; ++t) {
						const int tiled_col = tile_size*t+col;
						const int tiled_row = tile_size*t+row;
						for(int w = 0; w < works_per_thread; ++w) {
							a_sub[row][col+w*rts] =
								global_row < m_size && (tiled_col+w*rts) < k_size ?
									a[global_row*k_size+tiled_col+w*rts+a_offset] : 0.f;
							//b_sub[row][col] = tiled_row < k_size ?
								//b[global_col*k_size+tiled_row+b_offset] : 0.f;
							b_sub[row][col+w*rts] =
								tiled_row < k_size && (global_col+w*rts) < n_size ?
									b[tiled_row*n_size+global_col+w*rts+b_offset] : 0.f;
						}

						barrier(CLK_LOCAL_MEM_FENCE);

						for(int i = 0; i < tile_size; ++i) {
							for(int w = 0; w < works_per_thread; ++w) {
								sum[w] += a_sub[row][i]*b_sub[i][col+w*rts];
							}
						}

						barrier(CLK_LOCAL_MEM_FENCE);
					}
					for(int w = 0; w < works_per_thread; ++w) {
						if(global_col+w*rts < n_size && global_row < m_size) {
							c[global_row*n_size+global_col+w*rts+c_offset] = sum[w];
						}
					}
				}
				__kernel void matrix_multiply_more_works_per_thread2_reg(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 32;
					const int works_per_thread = 2;
					const int rts = tile_size/works_per_thread;
					const int lpt = works_per_thread*works_per_thread;

					const int tidn = get_local_id(0);
					const int tidm = get_local_id(1);
					const int offsetn = tile_size*get_group_id(0);
					const int offsetm = tile_size*get_group_id(1);
					__local float a_sub[32][32];
					__local float b_sub[32][32+2];
					float a_reg;
					float b_reg[2];
					float sum[2][2] = {};
					const int tile_num = (k_size-1)/tile_size+1;
					for(int t = 0; t < tile_num; ++t) {
						for(int l = 0; l < lpt; ++l) {
							const int tid = tidm*rts+tidn;
							const int id = l*rts*rts+tid;
							const int col = id%tile_size;
							const int row = id/tile_size;
							a_sub[row][col] =
								offsetm+row < m_size && tile_size*t+col < k_size ?
									a[(offsetm+row)*k_size+tile_size*t+col+a_offset] : 0.f;
									//a[tiled_index*k_size+offsetm+col+a_offset] : 0.f;
							b_sub[row][col] =
								tile_size*t+row < k_size && offsetn+col < n_size ?
									b[(tile_size*t+row)*n_size+offsetn+col+b_offset] : 0.f;
									//b[tiled_index*n_size+offsetn+col+b_offset] : 0.f;
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
								c[global_row*n_size+global_col+c_offset] = sum[wm][wn];
							}
						}
					}
				}
				__kernel void matrix_multiply_more_works_per_thread2_reg_64(
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
									b[(tile_size*t+row)*n_size+offsetn+col+b_offset] : 0.f;
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
								c[global_row*n_size+global_col+c_offset] = sum[wm][wn];
							}
						}
					}
				}
				__kernel void matrix_multiply_more_works_per_thread4_reg(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 32;
					const int works_per_thread = 4;
					const int rts = tile_size/works_per_thread;
					const int lpt = works_per_thread*works_per_thread;

					const int tidn = get_local_id(0);
					const int tidm = get_local_id(1);
					const int offsetn = tile_size*get_group_id(0);
					const int offsetm = tile_size*get_group_id(1);
					__local float a_sub[32][32];
					__local float b_sub[32][32+2];
					float a_reg;
					float b_reg[4];
					float sum[4][4] = {};
					const int tile_num = (k_size-1)/tile_size+1;
					for(int t = 0; t < tile_num; ++t) {
						for(int l = 0; l < lpt; ++l) {
							const int tid = tidm*rts+tidn;
							const int id = l*rts*rts+tid;
							const int col = id%tile_size;
							const int row = id/tile_size;
							a_sub[row][col] =
								offsetm+row < m_size && tile_size*t+col < k_size ?
									a[(offsetm+row)*k_size+tile_size*t+col+a_offset] : 0.f;
									//a[tiled_index*k_size+offsetm+col+a_offset] : 0.f;
							b_sub[row][col] =
								tile_size*t+row < k_size && offsetn+col < n_size ?
									b[(tile_size*t+row)*n_size+offsetn+col+b_offset] : 0.f;
									//b[tiled_index*n_size+offsetn+col+b_offset] : 0.f;
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
								c[global_row*n_size+global_col+c_offset] = sum[wm][wn];
							}
						}
					}
				}

				__kernel void matrix_multiply_more_works_per_thread4_reg_64(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 64;
					const int works_per_thread = 4;
					const int rts = tile_size/works_per_thread;
					const int lpt = works_per_thread*works_per_thread;

					const int tidn = get_local_id(0);
					const int tidm = get_local_id(1);
					const int offsetn = tile_size*get_group_id(0);
					const int offsetm = tile_size*get_group_id(1);
					__local float a_sub[64][64];
					__local float b_sub[64][64+2];
					float a_reg;
					float b_reg[4];
					float sum[4][4] = {};
					const int tile_num = (k_size-1)/tile_size+1;
					for(int t = 0; t < tile_num; ++t) {
						for(int l = 0; l < lpt; ++l) {
							const int tid = tidm*rts+tidn;
							const int id = l*rts*rts+tid;
							const int col = id%tile_size;
							const int row = id/tile_size;
							a_sub[row][col] =
								offsetm+row < m_size && tile_size*t+col < k_size ?
									a[(offsetm+row)*k_size+tile_size*t+col+a_offset] : 0.f;
									//a[tiled_index*k_size+offsetm+col+a_offset] : 0.f;
							b_sub[row][col] =
								tile_size*t+row < k_size && offsetn+col < n_size ?
									b[(tile_size*t+row)*n_size+offsetn+col+b_offset] : 0.f;
									//b[tiled_index*n_size+offsetn+col+b_offset] : 0.f;
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
								c[global_row*n_size+global_col+c_offset] = sum[wm][wn];
							}
						}
					}
				}
			);

			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_normal(
					neu::kernel& mm_kernel,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
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
				std::size_t global[2] = {4096, 4096};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, nullptr);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_tiled(
					neu::kernel& mm_kernel,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
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
				std::size_t global[2] = {4096, 4096};
				std::size_t local[2] = {32, 32};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_more_works_per_thread(
					neu::kernel& mm_kernel, int works_per_thread,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
				mm_kernel.set_args(
					static_cast<cl_int>(works_per_thread),
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
				std::size_t global[2] = {
					static_cast<std::size_t>(4096/works_per_thread),
					static_cast<std::size_t>(4096)};
				std::size_t local[2] = {
					static_cast<std::size_t>(32/works_per_thread),
					static_cast<std::size_t>(32)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_more_works_per_thread2(
					neu::kernel& mm_kernel,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
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
				std::size_t global[2] = {
					static_cast<std::size_t>(4096/2),
					static_cast<std::size_t>(4096)};
				std::size_t local[2] = {
					static_cast<std::size_t>(32/2),
					static_cast<std::size_t>(32)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_more_works_per_thread4(
					neu::kernel& mm_kernel,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
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
				std::size_t global[2] = {
					static_cast<std::size_t>(4096/4),
					static_cast<std::size_t>(4096)};
				std::size_t local[2] = {
					static_cast<std::size_t>(32/4),
					static_cast<std::size_t>(32)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_more_works_per_thread2_reg(
					neu::kernel& mm_kernel,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
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
				std::size_t global[2] = {
					static_cast<std::size_t>(4096/2),
					static_cast<std::size_t>(4096/2)};
				std::size_t local[2] = {
					static_cast<std::size_t>(32/2),
					static_cast<std::size_t>(32/2)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_more_works_per_thread2_reg_64(
					neu::kernel& mm_kernel,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
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
				std::size_t global[2] = {
					static_cast<std::size_t>(4096/2),
					static_cast<std::size_t>(4096/2)};
				std::size_t local[2] = {
					static_cast<std::size_t>(64/2),
					static_cast<std::size_t>(64/2)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_more_works_per_thread4_reg(
					neu::kernel& mm_kernel,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
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
				const std::size_t global[2] = {
					static_cast<std::size_t>(4096/4),
					static_cast<std::size_t>(4096/4)};
				const std::size_t local[2] = {
					static_cast<std::size_t>(32/4),
					static_cast<std::size_t>(32/4)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_more_works_per_thread4_reg_64(
					neu::kernel& mm_kernel,
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
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
				std::size_t global[2] = {
					static_cast<std::size_t>(4096/4),
					static_cast<std::size_t>(4096/4)};
				std::size_t local[2] = {
					static_cast<std::size_t>(64/4),
					static_cast<std::size_t>(64/4)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
		}
	}
}// namespace neu

#endif //NEU_LAYER_IMPL_COMMON_KERNEL_HPP
