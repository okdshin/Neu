#ifndef NEU_MAX_POOLING_LAYER_IMPL_HPP
#define NEU_MAX_POOLING_LAYER_IMPL_HPP
//20151026
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range_traits.hpp>
#include <neu/range_algorithm.hpp>
#include <neu/layer_traits.hpp>
#include <neu/kernel.hpp>
#include <neu/max_pooling_layer/kernel_source.hpp>

namespace neu {
	class max_pooling_layer {
	public:
		using layer_category = convolution_like_layer_tag;

		max_pooling_layer() = default;

		max_pooling_layer(std::size_t input_width, std::size_t output_width,
			std::size_t filter_width,
			std::size_t input_channel_num, std::size_t stride, std::size_t batch_size,
			kernel const& pooling_kernel,
			boost::compute::context const& context)
			: input_width_(input_width), filter_width_(filter_width),  
			input_channel_num_(input_channel_num),
			stride_(stride), batch_size_(batch_size),
			pooling_kernel_(pooling_kernel),
			output_width_(output_width),
			gpu_indices_(output_width*output_width*input_channel_num*batch_size, context) {}

		decltype(auto) input_width() const { return input_width_; }
		decltype(auto) input_channel_num() const { return input_channel_num_; }
		decltype(auto) output_width() const { return output_width_; }
		// output channel num is equal to the of input
		decltype(auto) output_channel_num() const { return input_channel_num_; }
		decltype(auto) batch_size() const { return batch_size_; }

		template<typename InputRange, typename OutputRange>
		decltype(auto) test_forward(std::size_t test_batch_size,
				InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			NEU_ASSERT(neu::range_distance(input) ==
				neu::layer_input_dim(*this)*test_batch_size);
			NEU_ASSERT(neu::range_distance(output) ==
				neu::layer_output_dim(*this)*test_batch_size);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
			enqueue_nd_range_kernel<3>(queue, pooling_kernel_,
				{0, 0, 0}, {output_width_, output_width_, test_batch_size},
				neu::range_get_buffer(input),
				static_cast<int>(neu::range_get_begin_index(input)),
				neu::range_get_buffer(output),
				static_cast<int>(neu::range_get_begin_index(output)),
				gpu_indices_,
				static_cast<int>(stride_), static_cast<int>(input_channel_num_),
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
		}

		template<typename InputRange, typename OutputRange>
		decltype(auto) backward(InputRange const& delta, OutputRange const& prev_delta,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			NEU_ASSERT(neu::range_distance(delta) == neu::layer_output_size(*this));
			NEU_ASSERT(neu::range_distance(prev_delta) == neu::layer_input_size(*this));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));

			cpu_indices indices(gpu_indices_.size());
			neu::range_copy(gpu_indices_, indices, queue);

			cpu_vector cpu_delta(neu::range_distance(delta));
			neu::range_copy(delta, cpu_delta, queue);

			cpu_vector cpu_prev_delta(neu::range_distance(prev_delta), 0.f);
			for(auto i = 0u; i < indices.size(); ++i) {
				NEU_ASSERT(i < indices.size());
				NEU_ASSERT(i < cpu_delta.size());
				NEU_ASSERT(static_cast<std::size_t>(indices[i]) < cpu_prev_delta.size());
				cpu_prev_delta[indices[i]] += cpu_delta[i];
			}
			neu::range_copy(cpu_prev_delta, prev_delta, queue);

			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
		}

	private:
		std::size_t input_width_;
		std::size_t filter_width_;
		std::size_t input_channel_num_;
		std::size_t stride_;
		std::size_t batch_size_;

		kernel pooling_kernel_;

		std::size_t output_width_;

		gpu_indices gpu_indices_;
	};
	decltype(auto) make_max_pooling_layer(
		std::size_t input_width, std::size_t output_width,
		std::size_t filter_width,
		std::size_t input_channel_num, std::size_t stride, std::size_t batch_size,
		boost::compute::context const& context
			=boost::compute::system::default_context()
	){
		auto pooling_kernel = make_kernel(max_pooling_kernel_source,
			"max_pooling", context);
		return max_pooling_layer(input_width, output_width, filter_width,
			input_channel_num, stride, batch_size, pooling_kernel,
			context);
	}
}// namespace neu

#endif //NEU_MAX_POOLING_LAYER_IMPL_HPP
