#ifndef NEU_VALIDATION_HPP
#define NEU_VALIDATION_HPP
//20151010
#include <boost/compute/algorithm/all_of.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, isfinite_kernel, (float x), {
		return isfinite(x);
	});
	decltype(auto) all_of_finite(gpu_vector const& x) {
		return boost::compute::all_of(x.begin(), x.end(), isfinite_kernel); 
	}
}// namespace neu

#endif //NEU_VALIDATION_HPP
