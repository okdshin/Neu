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

/*
BOOST_AUTO_TEST_CASE(matrix_multiply_test) {
	for(int m_size = 1; m_size < 64; ++m_size) {
	for(int n_size = 1; n_size < 64; ++n_size) {
	for(int k_size = 1; k_size < 64; ++k_size) {
	std::cout << m_size << " " << n_size << " " << k_size << std::endl;

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
		"matrix_multiply_more_works_per_thread",
		//"matrix_multiply_tiled",
		queue.get_context());
	neu::layer::impl::matrix_multiply_more_works_per_thread(
		mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
	neu::cpu_vector cpu_c(c.size());
	neu::range::copy(gpu_c, cpu_c, queue);
	for(int i = 0; i < static_cast<int>(c.size()); ++i) {
		//std::cout << c[i] << " " << gpu_c[i];
		BOOST_CHECK(std::abs(c[i]-cpu_c[i]) < 1.0e-3);
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
		std::cout << ((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
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
		std::cout << ((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
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
		std::cout << ((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
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
		std::cout << ((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
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
		std::cout << ((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
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
		std::cout << ((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
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
		std::cout << ((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
	{
		auto mm_kernel = neu::make_kernel(neu::layer::impl::common_kernel_source,
			"matrix_multiply_more_works_per_thread8_reg",
			queue.get_context());
		std::cout << "start more works per thread8 reg" << std::endl;
		boost::timer t;
		neu::layer::impl::matrix_multiply_more_works_per_thread8_reg(
			mm_kernel, gpu_a, gpu_b, gpu_c, m_size, n_size, k_size, queue);
		queue.finish();
		const auto elapsed = t.elapsed();
		std::cout << elapsed << std::endl;
		std::cout << ((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f << "GFLOPS" << std::endl;
		std::cout << 100.f*((4096.f*4096.f*4096.f)/elapsed)/1000'000'000.f/2308.f << "%" << std::endl;
	}
}
BOOST_AUTO_TEST_SUITE_END()
