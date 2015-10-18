#ifndef NEU_ACTIVATION_LAYER_HPP
#define NEU_ACTIVATION_LAYER_HPP
//20150901
#include <gsl.h>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/layer_parameter.hpp>
namespace neu {
	template<typename ActivationFunc, typename DiffActivationFunc>
	class activation_layer {
	public:
		activation_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			ActivationFunc const& activation_func,
			DiffActivationFunc const& diff_activation_func)
		: input_dim_(input_dim), output_dim_(output_dim), batch_size_(batch_size),
		activation_func_(activation_func), diff_activation_func_(diff_activation_func),
		input_(input_dim*batch_size), next_input_(output_dim*batch_size),
		prev_delta_(input_dim*batch_size) {}

		decltype(auto) forward(gpu_vector const& input) {
			Expects(is_all_of_finite(input));
			auto future = boost::compute::
				copy_async(input.begin(), input.end(), input_.begin());
			next_input_ = activation_func_(input);
			future.wait();
			Ensures(is_all_of_finite(next_input_));
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(gpu_vector const& delta) {
			Expects(is_all_of_finite(delta));
			auto df = diff_activation_func_(input_);
			Ensures(is_all_of_finite(df));
			boost::compute::transform(df.begin(), df.end(), delta.begin(),
				prev_delta_.begin(), boost::compute::multiplies<scalar>());
			Ensures(is_all_of_finite(prev_delta_));
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

	private:
		std::size_t input_dim_, output_dim_, batch_size_;
		ActivationFunc activation_func_;
		DiffActivationFunc diff_activation_func_;
		gpu_vector input_, next_input_, prev_delta_;
	};
	template<typename ActivationFunc>
	decltype(auto) make_activation_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			ActivationFunc const& activation_func) {
		return activation_layer<ActivationFunc, differential<ActivationFunc>>(
			input_dim, output_dim, batch_size,
			activation_func, differential<ActivationFunc>());
	}
	template<typename ActivationFunc>
	decltype(auto) make_activation_layer(
			std::size_t input_dim, std::size_t batch_size,
			ActivationFunc const& activation_func) {
		return activation_layer<ActivationFunc, differential<ActivationFunc>>(
			input_dim, input_dim, batch_size,
			activation_func, differential<ActivationFunc>());
	}

	class activation_layer_parameter {
		NEU_PP_PARAMETER(input_dim)
		NEU_PP_PARAMETER(output_dim)
		NEU_PP_PARAMETER(batch_size)
	};
	template<typename Param>
	decltype(auto) make_activation_layer_parameter(Param const& param) {
		activation_layer_parameter p;
		p.input_dim(param.output_dim());
		p.batch_size(param.batch_size());
		return p;
	}
	template<typename ActivationFunc>
	decltype(auto) make_activation_layer(
			activation_layer_parameter const& param,
			ActivationFunc const& activation_func) {
		auto output_dim = param.is_output_dim_set() ?
			param.output_dim() : param.input_dim();
		return activation_layer<ActivationFunc, differential<ActivationFunc>>(
			param.input_dim(), output_dim, param.batch_size(),
			activation_func, differential<ActivationFunc>());
	}
}// namespace neu

#endif //NEU_ACTIVATION_LAYER_HPP
