#ifndef NEU_SOFTMAX_WITH_LOSS_FUNCTION_LAYER_HPP
#define NEU_SOFTMAX_WITH_LOSS_FUNCTION_LAYER_HPP
//20150901
#include <gsl.h>
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/layer_parameter.hpp>
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
	template<typename LearningRateGen>
	class softmax_with_loss_function_layer {
	public:
		softmax_with_loss_function_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			kernel const& softmax_kernel)
		: input_dim_(input_dim), output_dim_(output_dim), batch_size_(batch_size),
		multiply_kernel_(softmax_kernel),
		input_(input_dim*batch_size), next_input_(output_dim*batch_size),
		prev_delta_(input_dim*batch_size) {
			if(weight.size() != input_dim*output_dim || bias.size() != output_dim) {
				throw std::invalid_argument(
					"the size of weight and/or bias are not correct.");
			}
		}

		decltype(auto) forward(gpu_vector const& input) {
			Expects(input.size() == input_dim_*batch_size_);
			Expects(is_all_of_finite(input));
			auto future = boost::compute::
				copy_async(input.begin(), input.end(), input_.begin());
			execute_nd_range_kernel<2>(multiply_kernel_,
				{0, 0}, {output_dim_, batch_size_},
				input, next_input_, weight_, bias_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_));
			future.wait();
			if(is_any_of_inf(next_input_)) {
				std::ofstream inputf("input.txt");
				std::ofstream weightf("weight.txt");
				std::ofstream biasf("bias.txt");
				std::ofstream next_inputf("next_input.txt");
				print(inputf, input, input_dim_);
				print(weightf, weight_, input_dim_);
				print(biasf, bias_, output_dim_);
				print(next_inputf, next_input_, output_dim_);
			}
			Ensures(!is_any_of_nan(next_input_));
			Ensures(!is_any_of_inf(next_input_));
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			Expects(delta.size() == output_dim_*batch_size_);
			Expects(is_all_of_finite(delta));
			auto future = boost::compute::
				copy_async(delta.begin(), delta.end(), delta_.begin());
			execute_nd_range_kernel<2>(multiply_back_kernel_,
				{0, 0}, {input_dim_, batch_size_},
				prev_delta_, delta, weight_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_));
			future.wait();
			/*
			if(!all_of_finite(prev_delta_)) {
				std::ofstream deltaf("delta.txt");
				std::ofstream weightf("weight.txt");
				std::ofstream biasf("bias.txt");
				std::ofstream prev_deltaf("prev_delta.txt");
				print(deltaf, delta);
				print(weightf, weight_);
				print(biasf, bias_);
				print(prev_deltaf, prev_delta_);
			}
			*/
			Ensures(is_all_of_finite(prev_delta_));
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

		decltype(auto) update() { //TODO async
			execute_nd_range_kernel<2>(calc_del_weight_kernel_,
				{0, 0}, {input_dim_, output_dim_},
				input_, delta_, del_weight_, del_bias_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_),
				static_cast<int>(batch_size_));
			Ensures(is_all_of_finite(del_weight_));
			Ensures(is_all_of_finite(del_bias_));
			learning_rate_gen_(weight_, bias_, del_weight_, del_bias_);
			Ensures(is_all_of_finite(weight_));
			Ensures(is_all_of_finite(bias_));
		}

		decltype(auto) get_weight() const { return (weight_); }
		decltype(auto) get_bias() const { return (bias_); }

	private:
		std::size_t input_dim_, output_dim_, batch_size_;
		gpu_vector weight_, bias_;
		LearningRateGen learning_rate_gen_;
		kernel multiply_kernel_, multiply_back_kernel_,	calc_del_weight_kernel_;
		gpu_vector input_, next_input_, delta_, prev_delta_;
		gpu_vector del_weight_, del_bias_;
	};
	template<typename LearningRateGen>
	decltype(auto) make_softmax_with_loss_function_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			gpu_vector const& weight, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen) {
		auto multiply_kernel
			= make_kernel(multiply_kernel_source, "multiply");
		auto multiply_back_kernel
			= make_kernel(multiply_back_kernel_source, "multiply_back");
		auto calc_del_weight_kernel
			= make_kernel(calc_del_weight_kernel_source, "calc_del_weight");
		return softmax_with_loss_function_layer<LearningRateGen>(
			input_dim, output_dim, batch_size, weight, bias, learning_rate_gen,
			multiply_kernel, multiply_back_kernel, calc_del_weight_kernel);
	}

	class softmax_with_loss_function_layer_parameter {
		NEU_PP_PARAMETER(input_dim)
		NEU_PP_PARAMETER(output_dim)
		NEU_PP_PARAMETER(batch_size)
	public:
		decltype(auto) weight_dim() const {
			return input_dim()*output_dim();
		}
		decltype(auto) bias_dim() const {
			return output_dim();
		}
	};
	template<typename Param>
	decltype(auto) make_softmax_with_loss_function_layer_parameter(Param const& param) {
		softmax_with_loss_function_layer_parameter p;
		p.input_dim(param.output_dim());
		p.batch_size(param.batch_size());
		return p;
	}
	template<typename LearningRateGen>
	decltype(auto) make_softmax_with_loss_function_layer(
			softmax_with_loss_function_layer_parameter const& param,
			gpu_vector const& weight, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen) {
		return make_softmax_with_loss_function_layer(
			param.input_dim(), param.output_dim(), param.batch_size(),
			weight, bias, learning_rate_gen);
	}
	template<typename WeightGen, typename BiasGen, typename LearningRateGen>
	decltype(auto) make_softmax_with_loss_function_layer(
			softmax_with_loss_function_layer_parameter const& param,
			WeightGen const& wg, BiasGen const& bg,
			LearningRateGen const& learning_rate_gen) {
		return make_softmax_with_loss_function_layer(
			param,
			neu::make_random_gpu_vector(param.weight_dim(), wg),
			neu::make_random_gpu_vector(param.bias_dim(), bg),
			learning_rate_gen);
	}
}// namespace neu

#endif //NEU_SOFTMAX_WITH_LOSS_FUNCTION_LAYER_HPP
