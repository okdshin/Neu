#ifndef NEU_VECTOR_IO_HPP
#define NEU_VECTOR_IO_HPP
//20150828
#include <iostream>
#include <fstream>
#include <algorithm>
#include <neu/assert.hpp>
#include<neu/basic_type.hpp>
#include<neu/range/traits.hpp>

namespace neu {
	/*
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
	decltype(auto) print(gpu_indices const& x, boost::compute::command_queue& queue) {
		std::cout << "[";
		boost::compute::copy(x.begin(), x.end(),
			std::ostream_iterator<scalar>(std::cout, ", "), queue);
		std::cout << "]" << std::endl;
	}
	*/

	template<typename Range>
	decltype(auto) print(std::ostream& os, Range const& range, std::size_t dim,
			boost::compute::command_queue& queue
				=boost::compute::system::default_queue()) {
		NEU_ASSERT(neu::range::distance(range)%dim == 0);
		auto line_num = neu::range::distance(range)/dim;
		auto first = neu::range::begin(range);
		os << "size: " << neu::range::distance(range) << std::endl;
		for(auto i = 0u; i < line_num; ++i) {
			os << "(";
			boost::compute::copy(first, first+dim,
				std::ostream_iterator<float>(os, ", "), queue);
			os << ")" << std::endl;
			first += dim;
		}
	}
}// namespace neu

#endif //NEU_VECTOR_IO_HPP
