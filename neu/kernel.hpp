#ifndef NEU_KERNEL_HPP
#define NEU_KERNEL_HPP
//20150625
#include <boost/compute/system.hpp>
#include <boost/compute/kernel.hpp>
namespace neu {
	using kernel = boost::compute::kernel;
	decltype(auto) make_kernel(const char* source, std::string const& name) {
		auto program = boost::compute::program::create_with_source(
			source, boost::compute::system::default_context());
		try {
			program.build();
		}
		catch(boost::compute::opencl_error& e) {
			std::cout << program.build_log() << std::endl;
		}
		return kernel(program, name.c_str());
	}
	template<std::size_t Dim, typename Kernel, typename... Args>
	decltype(auto) async_execute_nd_range_kernel(
		Kernel&& kernel,
		std::array<std::size_t, Dim> const& origin,
		std::array<std::size_t, Dim> const& region,
		Args&&... args
	) {
		static_assert(std::is_same<std::decay_t<Kernel>, boost::compute::kernel>::value,
			"Kernel must be boost::compute::kernel");
		kernel.set_args(std::forward<Args>(args)...);
		return boost::compute::system::default_queue().enqueue_nd_range_kernel(
			kernel, Dim, origin.data(), region.data(), nullptr);
	}
	template<std::size_t Dim, typename Kernel, typename... Args>
	decltype(auto) execute_nd_range_kernel(
		Kernel&& kernel,
		std::array<std::size_t, Dim> const& origin,
		std::array<std::size_t, Dim> const& region,
		Args&&... args
	) {
		async_execute_nd_range_kernel(
			std::forward<Kernel>(kernel),
			origin, region,
			std::forward<Args>(args)...)
		.wait();
	}
}// namespace neu

#endif //NEU_KERNEL_HPP
