#ifndef NEU_CONVOLUTION_LAYER_FACTORY_HPP
#define NEU_CONVOLUTION_LAYER_FACTORY_HPP
//20151005
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/convolution_layer/indices.hpp>
#include <neu/convolution_layer/kernel_source.hpp>
#include <neu/convolution_layer/convolution_layer_impl.hpp>
namespace neu {
	template<typename LearningRateGen>
	decltype(auto) make_convolution_layer(
		std::size_t input_width, std::size_t output_width,
		std::size_t filter_width,
		std::size_t input_channel_num, std::size_t output_channel_num,
		std::size_t stride, std::size_t pad, std::size_t batch_size,
		gpu_vector const& filters, gpu_vector const& bias,
		LearningRateGen const& learning_rate_gen
	) {
		auto indices =
			make_convolution_indices(input_width, output_width, filter_width, stride, pad);
		std::cout << "here" << std::endl;

		auto conv_kernel = make_kernel(convolution_kernel_source, "convolution");
		auto conv_back_kernel =
			make_kernel(convolution_back_kernel_source, "convolution_back");
		auto update_delta_filters_kernel
			= make_kernel(update_delta_filters_kernel_source, "update_delta_filters");
		return convolution_layer<LearningRateGen>(
			input_width, output_width, filter_width,
			input_channel_num, output_channel_num,
			stride, batch_size, filters, bias, learning_rate_gen, indices,
			conv_kernel, conv_back_kernel, update_delta_filters_kernel);
	}
}// namespace neu

#endif //NEU_CONVOLUTION_LAYER_FACTORY_HPP
