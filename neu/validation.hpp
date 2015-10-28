#ifndef NEU_VALIDATION_HPP
#define NEU_VALIDATION_HPP
//20151010
#include <neu/gpu_buffer_range.hpp>
#include <boost/compute/algorithm/all_of.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, isinf_kernel, (float x), {
		return isinf(x); //TODO
	});
	decltype(auto) is_any_of_inf(gpu_vector_range const& x) {
		return boost::compute::any_of(x.begin(), x.end(), isinf_kernel); 
	}
	BOOST_COMPUTE_FUNCTION(float, isnan_kernel, (float x), {
		return isnan(x); //TODO
	});
	decltype(auto) is_any_of_nan(gpu_vector_range const& x) {
		return boost::compute::any_of(x.begin(), x.end(), isnan_kernel); 
	}
	decltype(auto) is_any_of_nan(gpu_vector const& x) {
		return is_any_of_nan(to_range(x));
	}
	BOOST_COMPUTE_FUNCTION(float, isfinite_kernel, (float x), {
		return isfinite(x); //TODO
	});
	decltype(auto) is_all_of_finite(gpu_vector_range const& x) {
		return boost::compute::all_of(x.begin(), x.end(), isfinite_kernel); 
	}
	decltype(auto) is_all_of_finite(gpu_vector const& x) {
		return is_all_of_finite(to_range(x));
	}

	decltype(auto) l2_norm(gpu_vector const& x) {
		gpu_vector y(x.size());
		boost::compute::transform(x.begin(), x.end(), x.begin(),
			y.begin(), boost::compute::multiplies<scalar>());
		scalar sum = 0.f;
		boost::compute::reduce(y.begin(), y.end(), &sum, boost::compute::plus<scalar>());
		return sum;
	}

	decltype(auto) mean(gpu_vector const& x) {
		scalar sum = 0.f;
		boost::compute::reduce(x.begin(), x.end(), &sum, boost::compute::plus<scalar>());
		return sum/x.size();
	}

	decltype(auto) variance(gpu_vector const& x) {
		scalar sum = 0.f;
		boost::compute::reduce(x.begin(), x.end(), &sum, boost::compute::plus<scalar>());
		return sum/x.size();
	}
}// namespace neu

#endif //NEU_VALIDATION_HPP
