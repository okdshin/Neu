#ifndef NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
#define NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
//20150528
#include <cmath>
#include <gsl.h>
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/validation.hpp>
#include <neu/activation_func/derivative.hpp>
namespace neu {
	const char softmax_loss_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void softmax_loss(
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
	
	class softmax_loss {
	public:
		softmax_loss(std::size_t input_dim, std::size_t batch_size) :
			input_dim_(input_dim), batch_size_(batch_size),
			softmax_loss_kernel_(make_kernel(softmax_loss_kernel_source, "softmax_loss")),
			output_(input_dim*batch_size) {}
		decltype(auto) operator()(neu::gpu_vector const& input) {
			Expects(is_all_of_finite(input));
			Expects(output_.size() == input.size());
			/*
			std::ofstream inputf("softmax_loss_input.txt");
			std::cout << "input: "; print(inputf, x, input_dim_);
			*/
			//gpu_vector output(x.size());
			execute_nd_range_kernel<1>(softmax_loss_kernel_,
				{0}, {batch_size_}, input, output_, static_cast<int>(input_dim_));
			/*
			std::cout << "softmax_loss max element: " << *boost::compute::max_element(output.begin(), output.end()) << std::endl;
			std::cout << "softmax_loss min element: " << *boost::compute::min_element(output.begin(), output.end()) << std::endl;
			std::ofstream outputf("softmax_loss_output.txt");
			std::cout << "output: "; print(outputf, output, input_dim_);
			*/
			Ensures(!is_any_of_inf(output_));
			Ensures(!is_any_of_nan(output_));
			Ensures(boost::compute::all_of(output_.begin(), output_.end(),
				0 <= boost::compute::lambda::_1));
			Ensures(boost::compute::all_of(output_.begin(), output_.end(),
				boost::compute::lambda::_1 <= 1.f));
			return as_const(output_);
		}
	private:
		std::size_t input_dim_;
		std::size_t batch_size_;
		kernel softmax_loss_kernel_;
		gpu_vector output_;
	};
	template<>
	class derivative<softmax_loss> {
	public:
		derivative(std::size_t input_dim, std::size_t batch_size)
				: output_(input_dim*batch_size) {
			boost::compute::fill(output_.begin(), output_.end(), 1.f);
		}
	
		decltype(auto) operator()(neu::gpu_vector) const {
			return (output_);
		}
	private:
		gpu_vector output_;
	};
}// namespace neu

#endif //NEU_ACTIVATION_FUNC_SOFTMAX_LOSS_HPP
