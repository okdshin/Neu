#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/local_responce_normalization_layer.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	auto batch_size = 1u;
	neu::gpu_vector input;
	for(auto b = 0u; b < batch_size; ++b) {
		auto input_image = neu::load_rgb_image_as_3ch_image_vector("lena.bmp");
		auto cpu_input = std::get<0>(input_image);
		input.insert(input.end(), cpu_input.begin(), cpu_input.end());
	}
	auto input_width = 512u;
	auto channel_num = 3u;
	auto filter_width = 50u;
	auto output_width = input_width;

	/*
	auto lrn_across_maps = neu::make_local_responce_normalization_across_maps_layer(
		filter_width, 1.f, 0.5f, input_width, channel_num, batch_size);
	*/
	auto lrn_across_maps = neu::make_local_responce_normalization_same_map_layer(
		filter_width, 1.f, 0.5f, input_width, channel_num, batch_size);
	lrn_across_maps.forward(input);
	auto next_input = lrn_across_maps.get_next_input();
	lrn_across_maps.backward(neu::cpu_vector(next_input.size(), 1));
	auto prev_delta = lrn_across_maps.get_prev_delta();
	neu::save_image_vector_as_images(neu::to_cpu_vector(next_input),
		output_width, channel_num, batch_size, "output.bmp");
	neu::save_image_vector_as_images(neu::to_cpu_vector(prev_delta),
		input_width, channel_num, batch_size, "prev_delta.bmp");
}
