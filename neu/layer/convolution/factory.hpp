#ifndef NEU_CONVOLUTION_LAYER_FACTORY_HPP
#define NEU_CONVOLUTION_LAYER_FACTORY_HPP
//20151005
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/convolution_layer/indices.hpp>
#include <neu/convolution_layer/kernel_source.hpp>
#include <neu/convolution_layer/impl.hpp>
namespace neu {
	template<typename LearningRateGen>
	decltype(auto) make_convolution_layer(
		std::size_t input_width, std::size_t output_width,
		std::size_t filter_width,
		std::size_t input_channel_num, std::size_t output_channel_num,
		std::size_t stride, std::size_t pad, std::size_t batch_size,
		cpu_vector const& filters, cpu_vector const& bias,
		LearningRateGen const& learning_rate_gen,
		boost::compute::command_queue& queue
			=boost::compute::system::default_queue()
	) {
		auto indices = make_convolution_indices(
			input_width, output_width, filter_width, stride, pad);
		auto conv_kernel = make_kernel(convolution_kernel_source,
			"convolution", queue.get_context());
		auto conv_back_kernel = make_kernel(convolution_back_kernel_source,
			"convolution_back", queue.get_context());
		auto update_delta_filters_kernel = make_kernel(
			update_delta_filters_kernel_source, 
			"update_delta_filters", queue.get_context());
		auto filters_copy_queue = queue;
		gpu_vector f(filters.begin(), filters.end(), filters_copy_queue);
		auto bias_copy_queue = queue;
		gpu_vector b(bias.begin(), bias.end(), bias_copy_queue);
		filters_copy_queue.finish();
		bias_copy_queue.finish();
		return convolution_layer<LearningRateGen>(
			input_width, output_width, filter_width,
			input_channel_num, output_channel_num,
			stride, pad, batch_size,
			std::move(f), std::move(b),
			learning_rate_gen, indices,
			conv_kernel, conv_back_kernel, update_delta_filters_kernel,
			queue.get_context());
	}
}// namespace neu

#endif //NEU_CONVOLUTION_LAYER_FACTORY_HPP
