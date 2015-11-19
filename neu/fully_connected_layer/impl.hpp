#ifndef NEU_FULLY_CONNECTED_LAYER_IMPL_HPP
#define NEU_FULLY_CONNECTED_LAYER_IMPL_HPP
//20151023
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range_traits.hpp>
#include <neu/kernel.hpp>
#include <neu/fully_connected_layer/kernel_source.hpp>
#include <neu/learning_rate_gen.hpp>
namespace neu {
	template<typename LearningRateGen=learning_rate_gen>
	class fully_connected_layer {
	public:
		fully_connected_layer() = default;

		fully_connected_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			gpu_vector&& weight, gpu_vector&& bias,
			LearningRateGen const& learning_rate_gen,
			kernel const& multiply_kernel,
			kernel const& multiply_back_kernel,
			kernel const& calc_del_weight_kernel,
			boost::compute::context const& context) 
		: input_dim_(input_dim),
		output_dim_(output_dim),
		batch_size_(batch_size),
		weight_(std::move(weight)),
		bias_(std::move(bias)),
		learning_rate_gen_(learning_rate_gen),
		multiply_kernel_(multiply_kernel),
		multiply_back_kernel_(multiply_back_kernel),
		calc_del_weight_kernel_(calc_del_weight_kernel),
		input_(input_dim*batch_size, context),
		delta_(output_dim*batch_size, context),
		del_weight_(weight_.size(), context),
		del_bias_(bias_.size(), context) {
			if(weight_.size() != input_dim*output_dim || bias_.size() != output_dim) {
				throw std::invalid_argument(
					"the size of weight and/or bias are not correct.");
			}
		}

		decltype(auto) input_dim() const { return input_dim_; }
		decltype(auto) output_dim() const { return output_dim_; }
		decltype(auto) batch_size() const { return batch_size_; }
		//decltype(auto) weight() const { return to_cpu_vector(weight_); }
		//decltype(auto) bias() const { return to_cpu_vector(bias_); }
		decltype(auto) del_weight() const { return (del_weight_); }
		decltype(auto) del_bias() const { return (del_bias_); }
		decltype(auto) learning_rate_gen() const { return (learning_rate_gen_); }

		template<typename InputRange, typename OutputRange>
		decltype(auto) test_forward(std::size_t test_batch_size,
				InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range_distance(input) == input_dim_*test_batch_size);
			NEU_ASSERT(neu::range_distance(output) ==  output_dim_*test_batch_size);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
			enqueue_nd_range_kernel<2>(queue, multiply_kernel_,
				{0, 0}, {output_dim_, test_batch_size},
				neu::range_get_buffer(input),
				static_cast<cl_int>(neu::range_get_begin_index(input)),
				neu::range_get_buffer(output),
				static_cast<cl_int>(neu::range_get_begin_index(output)),
				weight_, bias_,
				static_cast<cl_int>(input_dim_), static_cast<cl_int>(output_dim_));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
		}

		template<typename InputRange, typename OutputRange>
		decltype(auto) forward(InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range_distance(input) == input_.size());
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
			neu::range_copy(input, input_, queue); //TODO
			test_forward(batch_size_, input, output, queue);
		}

		template<typename InputRange, typename OutputRange>
		decltype(auto) backward(InputRange const& delta, OutputRange const& prev_delta,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range_distance(delta) == delta_.size());
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
			neu::range_copy(delta, delta_, queue); //TODO
			enqueue_nd_range_kernel<2>(queue, multiply_back_kernel_,
				{0, 0}, {input_dim_, batch_size_},
				neu::range_get_buffer(prev_delta),
				static_cast<cl_int>(neu::range_get_begin_index(prev_delta)),
				neu::range_get_buffer(delta),
				static_cast<cl_int>(neu::range_get_begin_index(delta)),
				weight_,
				static_cast<cl_int>(input_dim_), static_cast<cl_int>(output_dim_));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
		}

		decltype(auto) update(boost::compute::command_queue& queue) {
			enqueue_nd_range_kernel<2>(queue, calc_del_weight_kernel_,
				{0, 0}, {input_dim_, output_dim_},
				input_, delta_, del_weight_, del_bias_,
				static_cast<cl_int>(input_dim_), static_cast<cl_int>(output_dim_),
				static_cast<cl_int>(batch_size_));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_weight_, queue));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_bias_, queue));
			learning_rate_gen_(weight_, bias_, del_weight_, del_bias_, queue);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(weight_, queue));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(bias_, queue));
		}

	private:
		std::size_t input_dim_, output_dim_, batch_size_;
		gpu_vector weight_, bias_;

		LearningRateGen learning_rate_gen_;
		kernel multiply_kernel_, multiply_back_kernel_,	calc_del_weight_kernel_;
		gpu_vector input_, delta_;
		gpu_vector del_weight_, del_bias_;
	};
	template<typename LearningRateGen>
	decltype(auto) make_fully_connected_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			cpu_vector const& weight, cpu_vector const& bias,
			LearningRateGen const& learning_rate_gen,
			boost::compute::command_queue& queue) {
		auto multiply_kernel = make_kernel(multiply_kernel_source,
			"multiply", queue.get_context());
		auto multiply_back_kernel = make_kernel(multiply_back_kernel_source,
			"multiply_back", queue.get_context());
		auto calc_del_weight_kernel = make_kernel(calc_del_weight_kernel_source,
			"calc_del_weight", queue.get_context());
		gpu_vector w(weight.begin(), weight.end(), queue);
		gpu_vector b(bias.begin(), bias.end(), queue);
		return fully_connected_layer<LearningRateGen>(
			input_dim, output_dim, batch_size,
			std::move(w), std::move(b),
			learning_rate_gen,
			multiply_kernel, multiply_back_kernel, calc_del_weight_kernel,
			queue.get_context());
	}
}// namespace neu

#endif //NEU_FULLY_CONNECTED_LAYER_IMPL_HPP
