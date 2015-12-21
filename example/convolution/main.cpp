#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/convolution/kernel.hpp>
#include <neu/learning_rate_gen/fixed_learning_rate_gen.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	auto input_image_path_list = std::vector<std::string>{
		//"../../../data/castle_bw.png",
		"../../../data/lena256.bmp"
	};
	auto batch_size = input_image_path_list.size();
	std::cout << "batch_size: " << batch_size << std::endl;

	neu::gpu_vector input;
	for(auto const& input_image_path : input_image_path_list) {
		auto input_image = neu::load_rgb_image_as_3ch_image_vector(input_image_path);
		auto cpu_input = std::get<0>(input_image);
		input.insert(input.end(), cpu_input.begin(), cpu_input.end());
	}

	auto input_width = 256u;
	auto input_channel_num = 3u;
	auto output_channel_num = 1u;
	auto filter_width = 11u;
	auto stride = 2u;
	auto pad = filter_width/2u;
	auto output_width = (input_width-filter_width+1u+2u*pad)/stride;
	std::cout << "output_width: " << output_width << std::endl;

	neu::gpu_vector filter;
	for(auto m = 0u; m < output_channel_num; ++m) {
		auto filter_image =
			neu::load_rgb_image_as_3ch_image_vector("../../../data/filter11.bmp");
		auto cpu_filter = std::get<0>(filter_image);
		filter.insert(filter.end(), cpu_filter.begin(), cpu_filter.end());
	}
	neu::gpu_vector bias(filter_width*filter_width*output_channel_num);
	boost::compute::fill(bias.begin(), bias.end(), 0.f);

	// Forward
	{
		neu::gpu_vector output(batch_size*output_channel_num*output_width*output_width);
		auto convolution_kernel = neu::make_kernel(
			neu::convolution_kernel_source, "convolution",
			boost::compute::system::default_context());
		auto ow = static_cast<decltype(batch_size)>(output_width);
		auto event = neu::enqueue_nd_range_kernel<3>(
			boost::compute::system::default_queue(),
			convolution_kernel,
			{0, 0, 0}, {ow, ow, batch_size},
			static_cast<int>(input_width), static_cast<int>(output_width),
			static_cast<int>(filter_width),
			static_cast<int>(input_channel_num), static_cast<int>(output_channel_num),
			static_cast<int>(stride), static_cast<int>(pad),
			neu::range_get_buffer(input),
			static_cast<int>(neu::range_get_begin_index(input)),
			neu::range_get_buffer(output),
			static_cast<int>(neu::range_get_begin_index(output)),
			filter, bias);
		event.wait();
		neu::save_image_vector_as_images(neu::to_cpu_vector(output),
			output_width, output_channel_num, batch_size, "output.bmp", 255.f);
	}

	// Backward
	{
		std::cout << "calc indices..." << std::flush;
		auto indices_tuple = neu::calc_convolution_indices(
			input_width, output_width, filter_width, stride, pad);
		auto indices_range_list = neu::to_gpu_indices(
			neu::make_range_list(std::get<0>(indices_tuple)));
		auto output_indices_list = neu::to_gpu_indices(
			neu::concat_indices(std::get<0>(indices_tuple)));
		auto filter_indices_list = neu::to_gpu_indices(
			neu::concat_indices(std::get<1>(indices_tuple)));
		neu::gpu_vector error(batch_size*output_channel_num*output_width*output_width);
		boost::compute::fill(error.begin(), error.end(), 1.);

		neu::gpu_vector delta(batch_size*input_channel_num*input_width*input_width);
		auto convolution_back_kernel = neu::make_kernel(
			neu::convolution_back_kernel_source, "convolution_back",
			boost::compute::system::default_context());
		auto event = neu::enqueue_nd_range_kernel<2>(
			boost::compute::system::default_queue(),
			convolution_back_kernel,
			{0, 0}, 
			{static_cast<decltype(batch_size)>(input_width*input_width), batch_size},
			indices_range_list, output_indices_list, filter_indices_list,
			static_cast<int>(input_width), static_cast<int>(output_width),
			static_cast<int>(filter_width),
			static_cast<int>(input_channel_num), static_cast<int>(output_channel_num),
			neu::range_get_buffer(delta),
			static_cast<int>(neu::range_get_begin_index(delta)),
			neu::range_get_buffer(error),
			static_cast<int>(neu::range_get_begin_index(error)),
			filter);
		event.wait();
		neu::save_image_vector_as_images(neu::to_cpu_vector(delta),
			input_width, input_channel_num, batch_size, "delta.bmp", 255.f);
	}
	/*
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
			filter_width, input_channel_num, output_channel_num, "delta_filters.bmp", 1.f);
	}
	*/
}
