#define BOOST_TEST_MODULE TestCommonKernel
#include <boost/test/unit_test.hpp>
#include <random>
#include <boost/timer.hpp>

#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/impl/common_kernel.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

decltype(auto) matrix_multiply_cpu(
		float const* a, float const* b, float* c,
		int m_size, int n_size, int k_size) {
	for (int m = 0; m < m_size; ++m) {
		for (int n = 0; n < n_size; ++n) {
			float acc = 0.f;
			for (int k = 0; k < k_size; ++k) {
				acc += a[m*k_size + k] * b[k*n_size + n];
			}
			c[m*n_size + n] = acc;
		}
	}
}
decltype(auto) matrix_transpose_cpu(
		float const* input, float* output, int p_size, int q_size) {
	for (int p = 0; p < p_size; ++p) {
		for (int q = 0; q < q_size; ++q) {
			output[q*p_size+p] = input[p*q_size+q];
		}
	}
}
BOOST_AUTO_TEST_CASE(matrix_transpose_test) {
	auto random_vector_gen = [](int size) {
		neu::cpu_vector vec(size);
		std::mt19937 rand(0);
		std::generate(vec.begin(), vec.end(),
			[&rand, dist=std::uniform_real_distribution<>(0.f, 10.f)]
			() mutable { return dist(rand); });
		return vec;
	};
	for(int p_size = 1; p_size < 128; ++p_size) {
	for(int q_size = 1; q_size < 128; ++q_size) {
		const auto input = random_vector_gen(p_size*q_size);
		neu::cpu_vector output(q_size*p_size, 0.f);
		matrix_transpose_cpu(input.data(), output.data(), p_size, q_size);

		const auto gpu_input = neu::to_gpu_vector(input, queue);
		neu::gpu_vector gpu_output(output.size(), 0.f, queue);

		neu::layer::impl::matrix_transpose(gpu_input, gpu_output, q_size, p_size, queue);
		/*
		{
			auto transpose_kernel =
				neu::make_kernel(neu::layer::impl::common_kernel_source,
				"matrix_transpose", queue.get_context());
			transpose_kernel.set_args(
				neu::range::get_buffer(gpu_input),
				static_cast<cl_int>(neu::range::get_begin_index(gpu_input)),
				neu::range::get_buffer(gpu_output),
				static_cast<cl_int>(neu::range::get_begin_index(gpu_output)),
				p_size, q_size);
			std::size_t global[2] = {
				static_cast<std::size_t>(((q_size-1)/32+1)*32),
				static_cast<std::size_t>(((p_size-1)/32+1)*32)
			};
			std::size_t local[2] = {
				static_cast<std::size_t>(32),
				static_cast<std::size_t>(32)
			};
			queue.enqueue_nd_range_kernel(transpose_kernel, 2, nullptr, global, local);
		}
		*/

		neu::cpu_vector cpu_output(output.size());
		neu::range::copy(gpu_output, cpu_output, queue);
		for(int i = 0; i < static_cast<int>(output.size()); ++i) {
			//std::cout << std::setprecision(10) << output[i] << " " << gpu_output[i] << "\n";
			BOOST_CHECK(std::abs(output[i]-cpu_output[i]) < 1.0e-3);
		}

	}}
}
/*
BOOST_AUTO_TEST_CASE(matrix_multiply_test) {
	for(int m_size = 1; m_size < 128; ++m_size) {
	for(int n_size = 1; n_size < 128; ++n_size) {
	for(int k_size = 1; k_size < 128; ++k_size) {

	auto random_vector_gen = [](int size) {
		neu::cpu_vector vec(size);
		std::mt19937 rand(0);
		std::generate(vec.begin(), vec.end(),
			[&rand, dist=std::uniform_real_distribution<>(0.f, 10.f)]
			() mutable { return dist(rand); });
		return vec;
	};
	const auto a = random_vector_gen(m_size*k_size);
	const auto b = random_vector_gen(k_size*n_size);
	neu::cpu_vector c(n_size*m_size, 0.f);
	matrix_multiply_cpu(a.data(), b.data(), c.data(), m_size, n_size, k_size);

	const auto gpu_a = neu::to_gpu_vector(a, queue);
	const auto gpu_b = neu::to_gpu_vector(b, queue);
	neu::gpu_vector gpu_c(c.size(), 0.f, queue);
	auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
		"matrix_multiply_more_works_per_thread2_reg_64",
		queue.get_context());
	neu::layer::impl::matrix_multiply_more_works_per_thread2_reg_64(
		mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
	neu::cpu_vector cpu_c(c.size());
	neu::range::copy(gpu_c, cpu_c, queue);
	queue.finish();
	for(int i = 0; i < static_cast<int>(c.size()); ++i) {
		if(std::abs(c[i]-cpu_c[i]) > 1.0e-3) {
			std::cout << m_size << " " << n_size << " " << k_size << std::endl;
			std::cout << std::setprecision(10) << c[i] << " " << gpu_c[i] << "\n";
			std::cout << c[i] - gpu_c[i] << "\n" << std::endl;
		}
	}

	}}}
}
*/
BOOST_AUTO_TEST_CASE(matrix_multiply_benchmark) {
	int m_size = 4096;
	int n_size = 4096;
	int k_size = 4096;
	auto random_vector_gen = [](int size) {
		neu::cpu_vector vec(size);
		std::mt19937 rand(0);
		std::generate(vec.begin(), vec.end(),
			[&rand, dist=std::uniform_real_distribution<>(0.f, 10.f)]
			() mutable { return dist(rand); });
		return vec;
	};
	const auto a = random_vector_gen(m_size*k_size);
	const auto b = random_vector_gen(k_size*n_size);
	neu::cpu_vector c(n_size*m_size, 0.f);
	//matrix_multiply_cpu(a.data(), b.data(), c.data(), m_size, n_size, k_size);

	const auto gpu_a = neu::to_gpu_vector(a, queue);
	const auto gpu_b = neu::to_gpu_vector(b, queue);
	neu::gpu_vector gpu_c(c.size(), 0.f, queue);
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_normal",
			queue.get_context());
		std::cout << "start normal" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_normal(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_tiled",
			queue.get_context());
		std::cout << "start tiled" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_tiled(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread2",
			queue.get_context());
		std::cout << "start more works per thread2" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread2(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread4",
			queue.get_context());
		std::cout << "start more works per thread4" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread2(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread2_non_bank_conflict",
			queue.get_context());
		std::cout << "start more works per thread2 non bank conflict" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread2(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread2_reg",
			queue.get_context());
		std::cout << "start more works per thread2 reg" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread2_reg(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread2_reg_64",
			queue.get_context());
		std::cout << "start more works per thread2 reg 64" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread2_reg_64(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread4_reg",
			queue.get_context());
		std::cout << "start more works per thread4 reg" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread4_reg(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread4_reg_64",
			queue.get_context());
		std::cout << "start more works per thread4 reg 64" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread4_reg_64(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread4_reg_128_rect",
			queue.get_context());
		std::cout << "start more works per thread4 reg 128 rect" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread4_reg_128_rect(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f*2.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
}
BOOST_AUTO_TEST_SUITE_END()
