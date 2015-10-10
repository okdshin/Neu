#ifndef NEU_ACTIVATION_FUNC_SOFTMAX_HPP
#define NEU_ACTIVATION_FUNC_SOFTMAX_HPP
//20150528
#include <cmath>
#include <gsl.h>
#include <neu/basic_type.hpp>
#include <neu/validation.hpp>
#include <neu/activation_func/differential.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, exp_kernel, (float x), {
		return exp(x);
	});
	class softmax {
	public:
		decltype(auto) operator()(neu::gpu_vector x) const {
			using boost::compute::lambda::_1;
			//std::cout << "x: "; print(x);
			x[0] = 0.f;
			boost::compute::transform(x.begin(), x.end(), x.begin(), exp_kernel);
			//std::cout << "exp(x): "; print(x);
			auto exp_sum = static_cast<scalar>(0.f);
			boost::compute::reduce(x.begin(), x.end(),
				&exp_sum, boost::compute::plus<scalar>());
			Ensures(std::isnormal(exp_sum));
			boost::compute::transform(x.begin(), x.end(), x.begin(), (_1/exp_sum));
			Ensures(all_of_finite(x));
			return x;
		}
	};
	template<>
	class differential<softmax> {
	public:
		//TODO
		decltype(auto) operator()(neu::gpu_vector const& x) const {
			return x;
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_SOFTMAX_HPP
