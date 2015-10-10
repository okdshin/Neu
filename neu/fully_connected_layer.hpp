#ifndef NEU_FULLY_CONNECTED_LAYER_HPP
#define NEU_FULLY_CONNECTED_LAYER_HPP
//20150901
#include <gsl.h>
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/layer_parameter.hpp>
namespace neu {
	const char multiply_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void multiply(
			const __global float* input, __global float* output,
			const __global float* weight, const __global float* bias,
			const int input_dim, const int output_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			float sum = bias[o];
			for(int i = 0; i < input_dim; ++i) {
				sum += weight[i+input_dim*o]*input[i+input_dim*b];
			}
			output[o+output_dim*b] = sum;
		}
	);
	const char multiply_back_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void multiply_back(
			__global float* input, const __global float* output,
			const __global float* weight,
			const int input_dim, const int output_dim)
		{
			const int b = get_global_id(1);
			const int i = get_global_id(0);

			float sum = 0.0;
			for(int o = 0; o < output_dim; ++o) {
				sum += weight[i+output_dim*o]*output[o+output_dim*b];
			}
			input[i+input_dim*b] = sum;
		}
	);
	const char calc_delta_weight_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void calc_delta_weight(
			const __global float* input, const __global float* delta,
			__global float* delta_weight, __global float* delta_bias,
			const int input_dim, const int output_dim, const int batch_size)
		{
			const int gr = get_global_id(1);
			const int gc = get_global_id(0);

			float weight_sum = 0.0;
			float bias_sum = 0.0;
			for(int b = 0; b < batch_size; ++b) {
				weight_sum += delta[gr+output_dim*b]*input[gc+input_dim*b];
				bias_sum += delta[gr+output_dim*b];
			}
			delta_weight[gc+input_dim*gr] = weight_sum/batch_size;
			delta_bias[gr] = bias_sum/batch_size;
		}
	);
	template<typename LearningRateGen>
	class fully_connected_layer {
	public:
		fully_connected_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			gpu_vector const& weight, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen,
			kernel const& multiply_kernel,
			kernel const& multiply_back_kernel,
			kernel const& calc_delta_weight_kernel) 
		: input_dim_(input_dim), output_dim_(output_dim), batch_size_(batch_size),
		weight_(weight), bias_(bias), learning_rate_gen_(learning_rate_gen),
		multiply_kernel_(multiply_kernel),
		multiply_back_kernel_(multiply_back_kernel),
		calc_delta_weight_kernel_(calc_delta_weight_kernel),
		input_(input_dim*batch_size), next_input_(output_dim*batch_size),
		delta_(output_dim*batch_size),
		prev_delta_(input_dim*batch_size),
		delta_weight_(weight.size()), delta_bias_(bias.size()) {
			if(weight.size() != input_dim*output_dim || bias.size() != output_dim) {
				throw std::invalid_argument(
					"the size of weight and/or bias are not correct.");
			}
		}

		decltype(auto) forward(gpu_vector const& input) {
			Expects(all_of_finite(input));
			auto future = boost::compute::
				copy_async(input.begin(), input.end(), input_.begin());
			execute_nd_range_kernel<2>(multiply_kernel_,
				{0, 0}, {output_dim_, batch_size_},
				input, next_input_, weight_, bias_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_));
			future.wait();
			Ensures(std::all_of(next_input_.begin(), next_input_.end(),
				[](auto e){ return std::isfinite(e); }));
			Ensures(all_of_finite(next_input_));
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			Expects(all_of_finite(delta));
			auto future = boost::compute::
				copy_async(delta.begin(), delta.end(), delta_.begin());
			execute_nd_range_kernel<2>(multiply_back_kernel_,
				{0, 0}, {input_dim_, batch_size_},
				prev_delta_, delta, weight_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_));
			future.wait();
			Ensures(all_of_finite(prev_delta_));
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

		decltype(auto) update() { //TODO async
			execute_nd_range_kernel<2>(calc_delta_weight_kernel_,
				{0, 0}, {input_dim_, output_dim_},
				input_, delta_, delta_weight_, delta_bias_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_),
				static_cast<int>(batch_size_));
			learning_rate_gen_(weight_, bias_, delta_weight_, delta_bias_);
			Ensures(all_of_finite(weight_));
			Ensures(all_of_finite(bias_));
		}

		decltype(auto) get_weight() const { return (weight_); }
		decltype(auto) get_bias() const { return (bias_); }

	private:
		std::size_t input_dim_, output_dim_, batch_size_;
		gpu_vector weight_, bias_;
		LearningRateGen learning_rate_gen_;
		kernel multiply_kernel_, multiply_back_kernel_,	calc_delta_weight_kernel_;
		gpu_vector input_, next_input_, delta_, prev_delta_;
		gpu_vector delta_weight_, delta_bias_;
	};
	template<typename LearningRateGen>
	decltype(auto) make_fully_connected_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			gpu_vector const& weight, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen) {
		auto multiply_kernel
			= make_kernel(multiply_kernel_source, "multiply");
		auto multiply_back_kernel
			= make_kernel(neu::multiply_back_kernel_source, "multiply_back");
		auto calc_delta_weight_kernel
			= make_kernel(calc_delta_weight_kernel_source, "calc_delta_weight");
		return fully_connected_layer<LearningRateGen>(
			input_dim, output_dim, batch_size, weight, bias, learning_rate_gen,
			multiply_kernel, multiply_back_kernel, calc_delta_weight_kernel);
	}

	class fully_connected_layer_parameter {
		NEU_PP_PARAMETER(input_dim)
		NEU_PP_PARAMETER(output_dim)
		NEU_PP_PARAMETER(batch_size)
	};
	template<typename Param>
	decltype(auto) make_fully_connected_layer_parameter(Param const& param) {
		fully_connected_layer_parameter p;
		p.input_dim(param.output_dim());
		p.batch_size(param.batch_size());
		return p;
	}
	template<typename LearningRateGen>
	decltype(auto) make_fully_connected_layer(
			fully_connected_layer_parameter const& param,
			gpu_vector const& weight, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen) {
		return make_fully_connected_layer(
			param.input_dim(), param.output_dim(), param.batch_size(),
			weight, bias, learning_rate_gen);
	}
	template<typename RandomNumberGenerator, typename LearningRateGen>
	decltype(auto) make_fully_connected_layer(
			fully_connected_layer_parameter const& param,
			RandomNumberGenerator const& g,
			LearningRateGen const& learning_rate_gen) {
		return make_fully_connected_layer(
			param,
			neu::make_random_gpu_vector(param.input_dim()*param.output_dim(), g),
			neu::make_random_gpu_vector(param.output_dim(), g),
			learning_rate_gen);
	}
}// namespace neu

#endif //NEU_FULLY_CONNECTED_LAYER_HPP
