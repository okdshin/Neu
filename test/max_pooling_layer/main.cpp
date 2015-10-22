#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/max_pooling_layer.hpp>
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
	auto filter_width = 3u;
	auto stride = 2u;
	auto output_width = input_width/stride;

	neu::max_pooling_layer_parameters params;
	params
	.input_width(512)
	.filter_width(3)
	.channel_num(3)
	.stride(2)
	.batch_size(1)
	;
	auto pool = neu::make_max_pooling_layer(params);

	/*
	auto pool = neu::make_max_pooling_layer(
		input_width, filter_width, channel_num, stride, batch_size);
	*/
	pool.forward(input);
	auto next_input = pool.get_next_input();
	pool.backward(neu::cpu_vector(next_input.size(), 1));
	auto prev_delta = pool.get_prev_delta();
	neu::save_image_vector_as_images(neu::to_cpu_vector(next_input),
		output_width, channel_num, batch_size, "output.bmp");
	neu::save_image_vector_as_images(neu::to_cpu_vector(prev_delta),
		input_width, channel_num, batch_size, "prev_delta.bmp");
}
