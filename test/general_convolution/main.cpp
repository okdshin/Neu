#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/convolution_layer.hpp>
#include <neu/learning_rate_gen/fixed_learning_rate_gen.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;
	auto input_image_path_list = std::vector<std::string>{
		"../../../data/lena256.bmp"
	};
	auto batch_size = input_image_path_list.size();

	neu::gpu_vector input;
	for(auto const& input_image_path : input_image_path_list) {
		auto input_image = neu::load_rgb_image_as_3ch_image_vector(input_image_path);
		auto cpu_input = std::get<0>(input_image);
		input.insert(input.end(), cpu_input.begin(), cpu_input.end());
	}

	auto input_width = 256u;
	auto input_channel_num = 3u;
	auto output_channel_num = 3u;
	auto filter_width = 11u;
	auto stride = 5u;
	auto pad = filter_width/2;
	auto output_width = (input_width+1)/stride;

	neu::gpu_vector filter;
	for(auto m = 0u; m < output_channel_num; ++m) {
		auto filter_image = neu::load_rgb_image_as_3ch_image_vector("../../../data/filter256.bmp");
		auto cpu_filter = std::get<0>(filter_image);
		filter.insert(filter.end(), cpu_filter.begin(), cpu_filter.end());
	}

	std::cout << "calc indices..." << std::flush;
	auto indices_tuple =
		neu::calc_convolution_indices(input_width, output_width, filter_width, stride, pad);
	std::cout << "finished" << std::endl;

	{
		auto indices_range_list = neu::to_gpu_indices(
			neu::make_range_list(std::get<0>(indices_tuple)));
		auto input_indices_list = neu::to_gpu_indices(
			neu::concat_indices(std::get<0>(indices_tuple)));
		auto filter_indices_list = neu::to_gpu_indices(
			neu::concat_indices(std::get<1>(indices_tuple)));
		neu::gpu_vector output(batch_size*output_channel_num*output_width*output_width);
		auto convolution_kernel = 
			neu::make_kernel(neu::convolution_kernel_source, "convolution");
		neu::execute_nd_range_kernel<2>(convolution_kernel,
			{0, 0},
			{static_cast<decltype(batch_size)>(output_width*output_width), batch_size},
			indices_range_list, input_indices_list, filter_indices_list,
			static_cast<int>(input_width), static_cast<int>(output_width),
			static_cast<int>(filter_width),
			static_cast<int>(input_channel_num), static_cast<int>(output_channel_num),
			input, output, filter);
		neu::save_image_vector_as_images(neu::to_cpu_vector(output),
			output_width, output_channel_num, batch_size, "output.bmp");
	}
	{
		neu::gpu_vector error(batch_size*output_channel_num*output_width*output_width);
		boost::compute::fill(error.begin(), error.end(), 1.);
		auto indices_range_list = neu::to_gpu_indices(
			neu::make_range_list(std::get<2>(indices_tuple)));
		auto output_indices_list = neu::to_gpu_indices(
			neu::concat_indices(std::get<2>(indices_tuple)));
		auto filter_indices_list = neu::to_gpu_indices(
			neu::concat_indices(std::get<3>(indices_tuple)));

		neu::gpu_vector delta(batch_size*input_channel_num*input_width*input_width);
		auto convolution_back_kernel = 
			neu::make_kernel(neu::convolution_back_kernel_source, "convolution_back");
		neu::execute_nd_range_kernel<2>(convolution_back_kernel,
			{0, 0}, 
			{static_cast<decltype(batch_size)>(input_width*input_width), batch_size},
			indices_range_list, output_indices_list, filter_indices_list,
			static_cast<int>(input_width), static_cast<int>(output_width),
			static_cast<int>(filter_width),
			static_cast<int>(input_channel_num), static_cast<int>(output_channel_num),
			delta, error, filter);
		neu::save_image_vector_as_images(neu::to_cpu_vector(delta),
			input_width, input_channel_num, batch_size, "delta.bmp");
	}
	{
		neu::gpu_vector error(batch_size*output_channel_num*output_width*output_width);
		boost::compute::fill(error.begin(), error.end(), 1.);
		auto indices_range_list = neu::to_gpu_indices(
			neu::make_range_list(std::get<4>(indices_tuple)));
		auto input_indices_list = neu::to_gpu_indices(
			neu::concat_indices(std::get<4>(indices_tuple)));
		auto output_indices_list = neu::to_gpu_indices(
			neu::concat_indices(std::get<5>(indices_tuple)));

		neu::gpu_vector delta_filters(
			input_channel_num*output_channel_num*filter_width*filter_width);
		auto update_delta_filters_kernel = 
			neu::make_kernel(neu::update_delta_filters_kernel_source, "update_delta_filters");
		neu::execute_nd_range_kernel<2>(update_delta_filters_kernel,
			{0, 0}, 
			{static_cast<decltype(output_channel_num)>(filter_width*filter_width), output_channel_num},
			indices_range_list, input_indices_list, output_indices_list,
			static_cast<int>(input_width), static_cast<int>(output_width),
			static_cast<int>(filter_width), static_cast<int>(batch_size),
			static_cast<int>(input_channel_num), static_cast<int>(output_channel_num),
			input, error, delta_filters);
		neu::save_image_vector_as_images(neu::to_cpu_vector(delta_filters),
			filter_width, input_channel_num, output_channel_num, "delta_filters.bmp");
	}
}
