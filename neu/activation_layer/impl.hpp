#ifndef NEU_ACTIVATION_LAYER_IMPL_HPP
#define NEU_ACTIVATION_LAYER_IMPL_HPP
//20151025
#include <neu/assert.hpp>
#include <neu/basic_type.hpp>
#include <neu/validation.hpp>
#include <neu/kernel.hpp>
#include <neu/activation_func/derivative.hpp>
#include <neu/range_algorithm.hpp>
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
		input_(input_dim*batch_size), df_(input_dim*batch_size) {}

		decltype(auto) input_dim() const { return input_dim_; }
		decltype(auto) output_dim() const { return output_dim_; }
		decltype(auto) batch_size() const { return batch_size_; }

		template<typename InputRange, typename OutputRange>
		decltype(auto) test_forward(std::size_t batch_size,
				InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue) {
			NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(input, queue));
			activation_func_(input, output, queue);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(output, queue));
		}

		template<typename InputRange, typename OutputRange>
		decltype(auto) forward(InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range_distance(input) == neu::range_distance(output));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(input, queue));
			neu::range_copy(input, input_, queue); //TODO
			activation_func_(input, output, queue);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(output, queue));
		}

		template<typename InputRange, typename OutputRange>
		decltype(auto) backward(InputRange const& delta, OutputRange const& prev_delta,
				boost::compute::command_queue& queue) {
			NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(delta, queue));
			diff_activation_func_(input_, df_, queue);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(df_, queue));
			neu::range_transform(df_, delta, prev_delta,
				boost::compute::multiplies<scalar>(), queue);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(prev_delta, queue));
		}

	private:
		std::size_t input_dim_, output_dim_, batch_size_;
		ActivationFunc activation_func_;
		DiffActivationFunc diff_activation_func_;
		gpu_vector input_, df_;
	};
	template<typename ActivationFunc>
	decltype(auto) make_activation_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			ActivationFunc const& activation_func) {
		return activation_layer<ActivationFunc, neu::derivative<ActivationFunc>>(
			input_dim, output_dim, batch_size,
			activation_func, neu::derivative<ActivationFunc>());
	}
}// namespace neu

#endif //NEU_ACTIVATION_LAYER_IMPL_HPP
