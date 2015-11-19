#ifndef NEU_CONVOLUTION_LAYER_IMPL_HPP
#define NEU_CONVOLUTION_LAYER_IMPL_HPP
//20151005
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range_traits.hpp>
#include <neu/range_algorithm.hpp>
#include <neu/layer_traits.hpp>
#include <neu/kernel.hpp>
#include <neu/convolution_layer/indices.hpp>

#include <chrono>
#include <thread>

namespace neu {
	template<typename LearningRateGen>
	class convolution_layer {
	public:
		using layer_category = convolution_like_layer_tag;
		convolution_layer(
			std::size_t input_width, std::size_t output_width,
			std::size_t filter_width,
			std::size_t input_channel_num, std::size_t output_channel_num,
			std::size_t stride, std::size_t pad, std::size_t batch_size,
			gpu_vector&& filters, gpu_vector&& bias,
			LearningRateGen const& learning_rate_gen,
			convolution_indices const& indices,
			kernel const& convolution_kernel,
			kernel const& convolution_back_kernel,
			kernel const& update_del_filters_kernel,
			boost::compute::context const& context)
		: input_width_(input_width), filter_width_(filter_width),
		input_channel_num_(input_channel_num), output_channel_num_(output_channel_num),
		stride_(stride), pad_(pad), batch_size_(batch_size),
		indices_(indices),
		filters_(std::move(filters)), bias_(std::move(bias)),
		learning_rate_gen_(learning_rate_gen),
		convolution_kernel_(convolution_kernel),
		convolution_back_kernel_(convolution_back_kernel),
		update_del_filters_kernel_(update_del_filters_kernel),
		output_width_(output_width),
		input_(input_width_*input_width_*input_channel_num_*batch_size_, context),
		delta_(output_width_*output_width_*output_channel_num_*batch_size_, context),
		del_filters_(filters_.size(), context),
		del_bias_(bias_.size(), context) {}

		decltype(auto) input_width() const { return input_width_; }
		decltype(auto) input_channel_num() const { return input_channel_num_; }
		decltype(auto) output_width() const { return output_width_; }
		decltype(auto) output_channel_num() const { return output_channel_num_; }
		decltype(auto) batch_size() const { return batch_size_; }

		decltype(auto) get_filters() const { return (filters_); }
		decltype(auto) get_bias() const { return (bias_); }

		decltype(auto) get_del_filters() const { return (del_filters_); }
		decltype(auto) get_del_bias() const { return (del_bias_); }

		decltype(auto) print_indices(std::ostream& os) {
			print(os, indices_.indices_range_list_for_input, indices_.indices_range_list_for_input.size());
			print(os, indices_.output_indices_list_for_input, indices_.output_indices_list_for_input.size());
			print(os, indices_.filter_indices_list_for_input, indices_.filter_indices_list_for_input.size());
		}

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
			neu::enqueue_nd_range_kernel<3>(queue, convolution_kernel_,
				{0, 0, 0},
				{output_width_, output_width_, test_batch_size},
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				static_cast<int>(stride_), static_cast<int>(pad_),
				neu::range_get_buffer(input),
				static_cast<int>(neu::range_get_begin_index(input)),
				neu::range_get_buffer(output),
				static_cast<int>(neu::range_get_begin_index(output)),
				filters_, bias_);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
		}
		template<typename InputRange, typename OutputRange>
		decltype(auto) forward(InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			NEU_ASSERT(neu::range_distance(input) == input_.size());
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
			neu::range_copy(input, input_, queue);
			test_forward(batch_size_, input, output, queue);
		}

		template<typename InputRange, typename OutputRange>
		decltype(auto) backward(InputRange const& delta, OutputRange const& prev_delta,
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			NEU_ASSERT(neu::range_distance(delta) == delta_.size());
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
			neu::range_copy(delta, delta_, queue);
			neu::enqueue_nd_range_kernel<2>(queue, convolution_back_kernel_,
				{0, 0}, {input_width_*input_width_, batch_size_},
				indices_.indices_range_list_for_input,
				indices_.output_indices_list_for_input,
				indices_.filter_indices_list_for_input,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				neu::range_get_buffer(prev_delta),
				static_cast<int>(neu::range_get_begin_index(prev_delta)),
				neu::range_get_buffer(delta),
				static_cast<int>(neu::range_get_begin_index(delta)),
				filters_);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
		}

		decltype(auto) update(
				boost::compute::command_queue& queue
					=boost::compute::system::default_queue()) {
			neu::enqueue_nd_range_kernel<3>(queue, update_del_filters_kernel_,
				{0, 0, 0}, {filter_width_, filter_width_, output_channel_num_},
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				static_cast<int>(output_channel_num_),
				static_cast<int>(stride_), static_cast<int>(pad_),
				static_cast<int>(batch_size_),
				input_, delta_, del_filters_, del_bias_);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_filters_, queue));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(del_bias_, queue));
			learning_rate_gen_(filters_, bias_, del_filters_, del_bias_, queue);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(filters_, queue));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(bias_, queue));
		}

	private:
		std::size_t input_width_;
		std::size_t filter_width_;
		std::size_t input_channel_num_;
		std::size_t output_channel_num_;
		std::size_t stride_;
		std::size_t pad_;
		std::size_t batch_size_;

		convolution_indices indices_;

		gpu_vector filters_;
		gpu_vector bias_;

		LearningRateGen learning_rate_gen_;

		kernel convolution_kernel_;
		kernel convolution_back_kernel_;
		kernel update_del_filters_kernel_;

		std::size_t output_width_;

		gpu_vector input_;
		gpu_vector delta_;

		gpu_vector del_filters_;
		gpu_vector del_bias_;
	};
}// namespace neu

#endif //NEU_CONVOLUTION_LAYER_IMPL_HPP
