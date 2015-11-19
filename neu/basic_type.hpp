#ifndef NEU_BASIC_TYPE_HPP
#define NEU_BASIC_TYPE_HPP
//20150619
#include<cassert>
#include<vector>
#include<boost/compute/container/vector.hpp>
#include<boost/compute/algorithm.hpp>
namespace neu {
	using scalar = cl_float;

	using cpu_vector = std::vector<scalar>;
	using cpu_indices = std::vector<cl_int>;

	using gpu_vector = boost::compute::vector<neu::scalar>;
	using gpu_indices = boost::compute::vector<cl_int>;

	template<typename Array>
	decltype(auto) to_gpu_vector(Array const& a) {
		using std::begin;
		using std::end;
		return neu::gpu_vector(begin(a), end(a));
	}
	template<typename Array>
	decltype(auto) to_gpu_indices(Array const& a) {
		using std::begin;
		using std::end;
		return neu::gpu_indices(begin(a), end(a));
	}

	decltype(auto) to_cpu_vector(gpu_vector const& x,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		cpu_vector cpu_x(x.size());
		boost::compute::copy(x.begin(), x.end(), cpu_x.begin(), queue);
		return cpu_x;
	}
	decltype(auto) to_cpu_indices(gpu_indices const& x,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		cpu_indices cpu_x(x.size());
		boost::compute::copy(x.begin(), x.end(), cpu_x.begin());
		return cpu_x;
	}

	template<typename Rand>
	decltype(auto) make_random_gpu_vector(std::size_t size, Rand const& rand) {
		cpu_vector cpu_vec(size);
		std::generate(cpu_vec.begin(), cpu_vec.end(), rand);
		return to_gpu_vector(cpu_vec);
	}
}// namespace neu

#endif //NEU_BASIC_TYPE_HPP
