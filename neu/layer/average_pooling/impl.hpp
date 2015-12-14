#ifndef NEU_AVERAGE_POOLING_LAYER_IMPL_HPP
#define NEU_AVERAGE_POOLING_LAYER_IMPL_HPP
//20151026
#include <neu/assert.hpp>
#include <neu/validation.hpp>
#include <neu/basic_type.hpp>
#include <neu/range_traits.hpp>
#include <neu/range_algorithm.hpp>
#include <neu/layer_traits.hpp>
#include <neu/kernel.hpp>
#include <neu/average_pooling_layer/kernel_source.hpp>
#include <neu/convolution_layer/indices.hpp>
namespace neu {
	class average_pooling_layer {
	public:
		using layer_category = convolution_like_layer_tag;

		average_pooling_layer() = default;

		average_pooling_layer(std::size_t input_width, std::size_t output_width,
			std::size_t filter_width,
			std::size_t input_channel_num,
			std::size_t stride, std::size_t pad, std::size_t batch_size,
			convolution_indices const& indices, gpu_vector const& filter,
			kernel const& pooling_kernel, kernel const& pooling_back_kernel)
			: input_width_(input_width), filter_width_(filter_width),  
			input_channel_num_(input_channel_num),
			stride_(stride), pad_(pad), batch_size_(batch_size),
			indices_(indices), filter_(filter),
			pooling_kernel_(pooling_kernel),
			pooling_back_kernel_(pooling_back_kernel),
			output_width_(output_width) {}

		decltype(auto) input_width() const { return input_width_; }
		decltype(auto) input_channel_num() const { return input_channel_num_; }
		decltype(auto) output_width() const { return output_width_; }
		// output channel num is equal to the of input
		decltype(auto) output_channel_num() const { return input_channel_num_; }
		decltype(auto) batch_size() const { return batch_size_; }

		template<typename InputRange, typename OutputRange>
		decltype(auto) test_forward(std::size_t test_batch_size,
				InputRange const& input, OutputRange const& output,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range_distance(input) ==
				neu::layer_input_dim(*this)*test_batch_size);
			NEU_ASSERT(neu::range_distance(output) ==
				neu::layer_output_dim(*this)*test_batch_size);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(input, queue));
			enqueue_nd_range_kernel<3>(queue, pooling_kernel_,
				{0, 0, 0}, {output_width_, output_width_, test_batch_size},
				static_cast<cl_int>(input_width_), static_cast<cl_int>(output_width_),
				static_cast<cl_int>(filter_width_),
				static_cast<cl_int>(input_channel_num_),
				static_cast<cl_int>(stride_), static_cast<cl_int>(pad_),
				neu::range_get_buffer(input),
				static_cast<cl_int>(neu::range_get_begin_index(input)),
				neu::range_get_buffer(output),
				static_cast<cl_int>(neu::range_get_begin_index(output)),
				filter_);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(output, queue));
		}

		template<typename InputRange, typename OutputRange>
		decltype(auto) backward(InputRange const& delta, OutputRange const& prev_delta,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(neu::range_distance(delta) == neu::layer_output_size(*this));
			NEU_ASSERT(neu::range_distance(prev_delta) == neu::layer_input_size(*this));
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(delta, queue));
			enqueue_nd_range_kernel<3>(queue, pooling_back_kernel_,
				{0, 0, 0}, {input_width_*input_width_, input_channel_num_, batch_size_},
				indices_.indices_range_list_for_input,
				indices_.output_indices_list_for_input,
				indices_.filter_indices_list_for_input,
				static_cast<int>(input_width_), static_cast<int>(output_width_),
				static_cast<int>(filter_width_),
				static_cast<int>(input_channel_num_),
				neu::range_get_buffer(prev_delta),
				static_cast<int>(neu::range_get_begin_index(prev_delta)),
				neu::range_get_buffer(delta),
				static_cast<int>(neu::range_get_begin_index(delta)),
				filter_);
			NEU_ASSERT_FOR_HEAVY_CALCULATION(is_all_of_finite(prev_delta, queue));
		}

	private:
		std::size_t input_width_;
		std::size_t filter_width_;
		std::size_t input_channel_num_;
		std::size_t stride_;
		std::size_t pad_;
		std::size_t batch_size_;

		convolution_indices indices_;

		gpu_vector filter_;

		kernel pooling_kernel_;
		kernel pooling_back_kernel_;

		std::size_t output_width_;
	};
	decltype(auto) make_average_pooling_layer(
		std::size_t input_width, std::size_t output_width, std::size_t filter_width,
		std::size_t input_channel_num, std::size_t stride, std::size_t pad,
		std::size_t batch_size, 
		gpu_vector const& filter,
		boost::compute::context const& context
			=boost::compute::system::default_context()
	){
		auto indices = neu::make_convolution_indices(
			input_width, output_width, filter_width, stride, pad);
		auto pooling_kernel = make_kernel(
			average_pooling_kernel_source, "average_pooling", context);
		auto pooling_back_kernel = make_kernel(
			average_pooling_back_kernel_source, "average_pooling_back", context);
		return average_pooling_layer(input_width, output_width, filter_width,
			input_channel_num, stride, pad, batch_size, indices, filter,
			pooling_kernel, pooling_back_kernel);
	}
}// namespace neu

#endif //NEU_AVERAGE_POOLING_LAYER_IMPL_HPP
