#ifndef NEU_LAYER_IMPL_COMMON_KERNEL_HPP
#define NEU_LAYER_IMPL_COMMON_KERNEL_HPP
//20160228
#include <boost/compute/utility/source.hpp>
#include <neu/range/traits.hpp>
namespace neu {
	namespace layer {
		namespace impl {
			constexpr char common_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
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

				__kernel void matrix_multiply_normal(
					const __global float* a, /*const int a_offset,*/
					const __global float* b, /*const int b_offset,*/
					__global float* c, /*const int c_offset,*/
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
				__kernel void matrix_multiply_more_works_per_thread4_reg_128_rect(
					const __global float* a, const int a_offset,
					const __global float* b, const int b_offset,
					__global float* c, const int c_offset,
					const int m_size, const int n_size, const int k_size)
				{
					const int tile_size = 128;
					const int tile_size_k = 16;
					const int works_per_thread = 4;
					const int rts = tile_size/works_per_thread;
					const int lpt = (tile_size_k*tile_size)/(rts*rts);

					const int tidn = get_local_id(0);
					const int tidm = get_local_id(1);
					const int offsetn = tile_size*get_group_id(0);
					const int offsetm = tile_size*get_group_id(1);
					__local float a_sub[16][128];
					__local float b_sub[128][16+2];
					float a_reg;
					float b_reg[4];
					float sum[4][4] = {};
					const int tile_num = (k_size-1)/tile_size_k+1;
					for(int t = 0; t < tile_num; ++t) {
						for(int l = 0; l < lpt; ++l) {
							const int tid = tidm*rts+tidn;
							volatile const int id = l*rts*rts+tid;
							const int col = id%tile_size;
							const int row = id/tile_size;
							a_sub[row][col] =
								offsetm+row < m_size && tile_size_k*t+col < k_size ?
									a[(offsetm+row)*k_size+tile_size_k*t+col+a_offset] : 0.f;
							b_sub[col][row] =
								tile_size_k*t+row < k_size && offsetn+col < n_size ?
									b[(tile_size_k*t+row)*n_size+offsetn+col+b_offset] : 0.f;
						}

						barrier(CLK_LOCAL_MEM_FENCE);

						for(int i = 0; i < tile_size_k; ++i) {
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
					//float a_reg;
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
								//a_reg = a_sub[row][i];
								for(int wn = 0; wn < works_per_thread; ++wn) {
									//sum[wm][wn] += a_reg*b_reg[wn];
									sum[wm][wn] += a_sub[row][i]*b_reg[wn];
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
					//static_cast<cl_int>(range::get_begin_index(a)),
					range::get_buffer(b),
					//static_cast<cl_int>(range::get_begin_index(b)),
					range::get_buffer(c),
					//static_cast<cl_int>(range::get_begin_index(c)),
					static_cast<cl_int>(m),
					static_cast<cl_int>(n),
					static_cast<cl_int>(k)
				);
				std::size_t global[2] = {4096, 4096};
				std::size_t local[2] = {32, 32};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
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
					static_cast<std::size_t>((n-1)/2+1 < 64 ? 64 : (n-1)/2+1),//TODO
					static_cast<std::size_t>((m-1)/2+1 < 64 ? 64 : (m-1)/2+1) //TODO
				};
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
				std::size_t global[2] = {
					static_cast<std::size_t>((n-1)/4+1 < 32 ? 32 : (n-1)/4+1),
					static_cast<std::size_t>((m-1)/4+1 < 32 ? 32 : (m-1)/4+1)
				};
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
					static_cast<std::size_t>((n-1)/4+1 < 64 ? 64 : (n-1)/4+1),
					static_cast<std::size_t>((m-1)/4+1 < 64 ? 64 : (m-1)/4+1)
				};
				std::size_t local[2] = {
					static_cast<std::size_t>(64/4),
					static_cast<std::size_t>(64/4)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply_more_works_per_thread4_reg_128_rect(
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
					static_cast<std::size_t>((n-1)/4+1 < 128 ? 128 : (n-1)/4+1),
					static_cast<std::size_t>((m-1)/4+1 < 128 ? 128 : (m-1)/4+1)
				};
				std::size_t local[2] = {
					static_cast<std::size_t>(128/4),
					static_cast<std::size_t>(128/4)};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}

			//
			// for gtx960, gtx980
			//
			template<typename RangeA, typename RangeB, typename RangeC>
			decltype(auto) matrix_multiply(
					RangeA const& a, RangeB const& b, RangeC& c,
					int m, int n, int k,
					boost::compute::command_queue& queue) {
				static auto mm_kernel = neu::make_kernel(
					neu::layer::impl::common_kernel_source,
					"matrix_multiply_more_works_per_thread2_reg_64",
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
				std::size_t global[2] = {
					static_cast<std::size_t>(((n-1)/64+1)*64),
					static_cast<std::size_t>(((m-1)/64+1)*64)
				};
				std::size_t local[2] = {
					static_cast<std::size_t>(64/2),
					static_cast<std::size_t>(64/2)
				};
				queue.enqueue_nd_range_kernel(mm_kernel, 2, nullptr, global, local);
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) matrix_transpose(
					InputRange const& input, OutputRange& output,
					int row_size, int col_size,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(row_size*col_size == range::distance(input));
				auto transpose_kernel =
					neu::make_kernel(neu::layer::impl::common_kernel_source,
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

#endif //NEU_LAYER_IMPL_COMMON_KERNEL_HPP
