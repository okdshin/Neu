#ifndef NEU_VALIDATION_HPP
#define NEU_VALIDATION_HPP
//20151010
#include <neu/range/gpu_buffer_range.hpp>
#include <boost/compute/algorithm/all_of.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, isinf_kernel, (float x), {
		return isinf(x);
	});
	template<typename Range>
	decltype(auto) is_any_of_inf(Range const& range) {
		return boost::compute::any_of(
			range::begin(range), range::end(range), isinf_kernel); 
	}

	BOOST_COMPUTE_FUNCTION(float, isnan_kernel, (float x), {
		return isnan(x);
	});
	template<typename Range>
	decltype(auto) is_any_of_nan(Range const& range) {
		return boost::compute::any_of(
			range::begin(range), range::end(range), isnan_kernel); 
	}

	BOOST_COMPUTE_FUNCTION(float, isfinite_kernel, (float x), {
		return isfinite(x);
	});
	template<typename Range>
	decltype(auto) is_all_of_finite(Range const& range,
			boost::compute::command_queue& queue) {
		return boost::compute::all_of(
			range::begin(range), range::end(range), isfinite_kernel, queue); 
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

	template<typename F>
	decltype(auto) calc_analytic_gradient(F&& f, scalar theta, scalar eps) {
		return (f(theta+eps)-f(theta-eps))/(2*eps);
	}

	template<typename T>
	decltype(auto) calc_relative_error(T lhs, T rhs) {
		return std::abs(lhs-rhs)
			/std::max<T>({std::abs(lhs), std::abs(rhs), static_cast<T>(1)});
	}

}// namespace neu

#endif //NEU_VALIDATION_HPP
