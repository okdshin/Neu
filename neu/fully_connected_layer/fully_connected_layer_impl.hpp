#ifndef NEU_FULLY_CONNECTED_LAYER_FULLY_CONNECTED_LAYER_IMPL_HPP
#define NEU_FULLY_CONNECTED_LAYER_FULLY_CONNECTED_LAYER_IMPL_HPP
//20151023
#include <gsl.h>
#include <neu/basic_type.hpp>
#include <neu/validation.hpp>
#include <neu/kernel.hpp>
#include <neu/fully_connected_layer/kernel_source.hpp>
#include <neu/learning_rate_gen.hpp>
namespace neu {
	template<typename LearningRateGen=learning_rate_gen>
	class fully_connected_layer {
	public:
		fully_connected_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			gpu_vector const& weight, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen,
			kernel const& multiply_kernel,
			kernel const& multiply_back_kernel,
			kernel const& calc_del_weight_kernel) 
		: input_dim_(input_dim), output_dim_(output_dim), batch_size_(batch_size),
		weight_(weight), bias_(bias), learning_rate_gen_(learning_rate_gen),
		multiply_kernel_(multiply_kernel),
		multiply_back_kernel_(multiply_back_kernel),
		calc_del_weight_kernel_(calc_del_weight_kernel),
		input_(input_dim*batch_size), next_input_(output_dim*batch_size),
		delta_(output_dim*batch_size),
		prev_delta_(input_dim*batch_size),
		del_weight_(weight.size()), del_bias_(bias.size()) {
			if(weight.size() != input_dim*output_dim || bias.size() != output_dim) {
				throw std::invalid_argument(
					"the size of weight and/or bias are not correct.");
			}
		}

		decltype(auto) input_dim() const { return input_dim_; }
		decltype(auto) output_dim() const { return output_dim_; }
		decltype(auto) batch_size() const { return batch_size_; }
		decltype(auto) weight() const { return to_cpu_vector(weight_); }
		decltype(auto) bias() const { return to_cpu_vector(bias_); }
		decltype(auto) learning_rate_gen() const { return (learning_rate_gen_); }

		decltype(auto) forward(gpu_vector const& input) {
			Expects(input.size() == input_dim_*batch_size_);
			Expects(is_all_of_finite(input));
			input_ = input;
			auto event = enqueue_nd_range_kernel<2>(multiply_kernel_,
				{0, 0}, {output_dim_, batch_size_},
				input, next_input_, weight_, bias_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_));
			event.wait();
			Ensures(!is_any_of_nan(next_input_));
			Ensures(!is_any_of_inf(next_input_));
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			Expects(delta.size() == output_dim_*batch_size_);
			Expects(is_all_of_finite(delta));
			delta_ = delta;
			auto = enqueue_nd_range_kernel<2>(multiply_back_kernel_,
				{0, 0}, {input_dim_, batch_size_},
				prev_delta_, delta, weight_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_));
			event.wait();
			Ensures(is_all_of_finite(prev_delta_));
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

		decltype(auto) update() {
			auto event = enqueue_nd_range_kernel<2>(calc_del_weight_kernel_,
				{0, 0}, {input_dim_, output_dim_},
				input_, delta_, del_weight_, del_bias_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_),
				static_cast<int>(batch_size_));
			event.wait();
			Ensures(is_all_of_finite(del_weight_));
			Ensures(is_all_of_finite(del_bias_));
			learning_rate_gen_(weight_, bias_, del_weight_, del_bias_);
			Ensures(is_all_of_finite(weight_));
			Ensures(is_all_of_finite(bias_));
		}

	private:
		std::size_t input_dim_, output_dim_, batch_size_;
		gpu_vector weight_, bias_;

		LearningRateGen learning_rate_gen_;
		kernel multiply_kernel_, multiply_back_kernel_,	calc_del_weight_kernel_;
		gpu_vector input_, next_input_, delta_, prev_delta_;
		gpu_vector del_weight_, del_bias_;
	};
	template<typename LearningRateGen>
	decltype(auto) make_fully_connected_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			gpu_vector const& weight, gpu_vector const& bias,
			LearningRateGen const& learning_rate_gen) {
		auto multiply_kernel
			= make_kernel(multiply_kernel_source, "multiply");
		auto multiply_back_kernel
			= make_kernel(multiply_back_kernel_source, "multiply_back");
		auto calc_del_weight_kernel
			= make_kernel(calc_del_weight_kernel_source, "calc_del_weight");
		return fully_connected_layer<LearningRateGen>(
			input_dim, output_dim, batch_size, weight, bias, learning_rate_gen,
			multiply_kernel, multiply_back_kernel, calc_del_weight_kernel);
	}
}// namespace neu

#endif //NEU_FULLY_CONNECTED_LAYER_FULLY_CONNECTED_LAYER_IMPL_HPP
