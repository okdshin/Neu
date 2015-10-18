#ifndef NEU_ACTIVATION_FUNC_SOFTMAX_HPP
#define NEU_ACTIVATION_FUNC_SOFTMAX_HPP
//20150528
#include <cmath>
#include <gsl.h>
#include <boost/compute/functional/bind.hpp>
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/validation.hpp>
#include <neu/activation_func/differential.hpp>
namespace neu {
	const char softmax_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void softmax(
			__global float* input, __global float* output,
			const int input_dim)
		{
			const int b = get_global_id(0);
			float m = input[0+b*input_dim];
			for(int i = 0; i < input_dim; ++i) {
				if(m < input[i+b*input_dim]) {
					m = input[i+b*input_dim];
				}
			}
			for(int i = 0; i < input_dim; ++i) {
				input[i+b*input_dim] -= m;
			}

			float sum = 0.f;
			for(int i = 0; i < input_dim; ++i) {
				sum += exp(input[i+b*input_dim]);
			}
			const float log_sum = log(sum);
			for(int i = 0; i < input_dim; ++i) {
				output[i+b*input_dim] = exp(input[i+b*input_dim]-log_sum);
				//output[i+b*input_dim] = exp(input[i+b*input_dim])/sum;
				
			}
		}
	);
	
	class softmax {
	public:
		softmax(std::size_t input_dim, std::size_t batch_size) :
			input_dim_(input_dim), batch_size_(batch_size),
			softmax_kernel_(make_kernel(softmax_kernel_source, "softmax")) {}
		decltype(auto) operator()(neu::gpu_vector x) {
			Expects(is_all_of_finite(x));
			std::ofstream inputf("softmax_input.txt");
			std::cout << "input: "; print(inputf, x, input_dim_);
			gpu_vector output(x.size());
			execute_nd_range_kernel<1>(softmax_kernel_,
				{0}, {batch_size_},
				x, output, static_cast<int>(input_dim_));
			std::cout << "softmax max element: " << *boost::compute::max_element(output.begin(), output.end()) << std::endl;
			std::cout << "softmax min element: " << *boost::compute::min_element(output.begin(), output.end()) << std::endl;
			std::ofstream outputf("softmax_output.txt");
			std::cout << "output: "; print(outputf, output, input_dim_);
			Ensures(!is_any_of_inf(output));
			Ensures(!is_any_of_nan(output));
			Ensures(boost::compute::all_of(output.begin(), output.end(),
				0 <= boost::compute::lambda::_1));
			Ensures(boost::compute::all_of(output.begin(), output.end(),
				boost::compute::lambda::_1 <= 1.f));
			return output;
		}
	private:
		std::size_t input_dim_;
		std::size_t batch_size_;
		kernel softmax_kernel_;
	};
	template<>
	class differential<softmax> {
	public:
		decltype(auto) operator()(neu::gpu_vector x) const {
			boost::compute::fill(x.begin(), x.end(), 1.);
			boost::compute::system::default_queue().finish();
			return x;
		}
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_SOFTMAX_HPP
