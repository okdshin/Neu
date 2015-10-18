#ifndef NEU_VECTOR_IO_HPP
#define NEU_VECTOR_IO_HPP
//20150828
#include <gsl.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include<neu/basic_type.hpp>

namespace neu {
	decltype(auto) print(cpu_vector const& x) {
		std::cout << "(";
		std::copy(x.begin(), x.end(),
			std::ostream_iterator<scalar>(std::cout, ", "));
		std::cout << ")" << std::endl;
	}
	decltype(auto) print(gpu_vector const& x) {
		std::cout << "[";
		boost::compute::copy(x.begin(), x.end(),
			std::ostream_iterator<scalar>(std::cout, ", "));
		std::cout << "]" << std::endl;
	}
	decltype(auto) print(cpu_indices const& x) {
		std::cout << "(";
		std::copy(x.begin(), x.end(),
			std::ostream_iterator<std::size_t>(std::cout, ", "));
		std::cout << ")" << std::endl;
	}
	decltype(auto) print(gpu_indices const& x) {
		std::cout << "[";
		boost::compute::copy(x.begin(), x.end(),
			std::ostream_iterator<scalar>(std::cout, ", "));
		std::cout << "]" << std::endl;
	}
	decltype(auto) print(std::ofstream& ofs, gpu_vector const& x, std::size_t dim) {
		Expects(x.size()%dim == 0);
		auto line_num = x.size()/dim;
		auto first = x.begin();
		ofs << "size: " << x.size() << std::endl;
		for(auto i = 0u; i < line_num; ++i) {
			ofs << "(";
			boost::compute::copy(first, first+dim, std::ostream_iterator<float>(ofs, ", "));
			ofs << ")" << std::endl;
			first += dim;
		}
	}
}// namespace neu

#endif //NEU_VECTOR_IO_HPP
