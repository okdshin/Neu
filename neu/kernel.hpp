#ifndef NEU_KERNEL_HPP
#define NEU_KERNEL_HPP
//20150625
#include <iostream>
#include <boost/compute/kernel.hpp>
#include <boost/compute/command_queue.hpp>
namespace neu {
	using kernel = boost::compute::kernel;
	decltype(auto) make_kernel(const char* source, std::string const& name,
			boost::compute::context const& context) {
		auto program = boost::compute::program::create_with_source(source, context);
		try {
			program.build();
		}
		catch(boost::compute::opencl_error& e) {
			std::cout << program.build_log() << std::endl;
		}
		return kernel(program, name.c_str());
	}
	template<int Dim, typename... Args>
	decltype(auto) enqueue_nd_range_kernel(
		boost::compute::command_queue& queue,
		kernel& kernel,
		std::array<std::size_t, Dim> const& origin,
		std::array<std::size_t, Dim> const& region,
		Args&&... args
	) {
		kernel.set_args(std::forward<Args>(args)...);
		return queue.enqueue_nd_range_kernel(
			kernel,
			static_cast<std::size_t>(Dim),
			origin.data(),
			region.data(),
			nullptr);
	}
}// namespace neu

#endif //NEU_KERNEL_HPP
