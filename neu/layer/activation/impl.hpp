#ifndef NEU_LAYER_ACTIVATION_IMPL_HPP
#define NEU_LAYER_ACTIVATION_IMPL_HPP
//20151025
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/activation_func/derivative.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/layer/traits.hpp>
namespace neu {
	namespace layer {
		template<typename ActivationFunc,
			typename DerivativeActivationFunc=neu::derivative<ActivationFunc>>
		class activation {
		public:
			using activation_func_type = ActivationFunc;
			using derivative_activation_func_type = DerivativeActivationFunc;

			activation() = default;
			activation(activation const&) = default;
			activation& operator=(activation const&) = default;
			activation(activation&&) = default;
			activation& operator=(activation&&) = default;
			~activation() = default;

			activation(
				std::size_t input_dim, std::size_t batch_size,
				activation_func_type const& activation_func,
				derivative_activation_func_type const& derivative_activation_func)
			: input_dim_(input_dim), batch_size_(batch_size),
			activation_func_(activation_func),
			derivative_activation_func_(derivative_activation_func),
			input_(input_dim*batch_size), df_(input_dim*batch_size) {}

			activation(std::size_t input_dim, std::size_t batch_size)
			: activation(input_dim, batch_size,
				activation_func_type(),
				derivative_activation_func_type()) {}

			decltype(auto) input_dim() const { return input_dim_; }
			decltype(auto) output_dim() const { return input_dim_; }
			decltype(auto) batch_size() const { return batch_size_; }
			
			decltype(auto) activation_func() const { return activation_func_; }

			template<typename InputRange, typename OutputRange>
			decltype(auto) test_forward(std::size_t batch_size,
					InputRange const& input, OutputRange const& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(input, queue));
				activation_func_(input, output, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(output, queue));
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) forward(
					InputRange const& input, OutputRange const& output,
					boost::compute::command_queue& queue) {
				NEU_ASSERT(range::distance(input) == range::distance(output));
				NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(input, queue));
				range::copy(input, input_, queue); //TODO async operation
				activation_func_(input, output, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(output, queue));
			}

			template<typename InputRange, typename OutputRange>
			decltype(auto) backward(
					InputRange const& delta, OutputRange const& prev_delta,
					bool is_top,
					boost::compute::command_queue& queue) {
				NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(delta, queue));
				derivative_activation_func_(input_, df_, queue);
				NEU_ASSERT_FOR_HEAVY_CALCULATION(neu::is_all_of_finite(df_, queue));
				if(!is_top) {
					range::transform(df_, delta, prev_delta,
						boost::compute::multiplies<scalar>(), queue);
				}
				NEU_ASSERT_FOR_HEAVY_CALCULATION(
					neu::is_all_of_finite(prev_delta, queue));
			}

			decltype(auto) update(boost::compute::command_queue&) { /*do nothing*/ }

		private:
			std::size_t input_dim_, batch_size_;
			activation_func_type activation_func_;
			derivative_activation_func_type derivative_activation_func_;
			gpu_vector input_, df_;
		};
		template<typename ActivationFunc, typename DerivativeActivationFunc>
		decltype(auto) make_activation(
				std::size_t input_dim, std::size_t batch_size,
				ActivationFunc const& activation_func,
				DerivativeActivationFunc const& derivative_activation_func) {
			return activation<ActivationFunc, DerivativeActivationFunc>(
				input_dim, batch_size, activation_func, derivative_activation_func);
		}
		template<typename ActivationFunc>
		decltype(auto) make_activation(
				std::size_t input_dim, std::size_t batch_size,
				ActivationFunc const& activation_func) {
			return make_activation(input_dim, batch_size,
				activation_func, neu::derivative<ActivationFunc>());
		}
	}
}// namespace neu

#endif //NEU_LAYER_ACTIVATION_IMPL_HPP
