#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/average_pooling_layer.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	auto batch_size = 1u;
	neu::gpu_vector input;
	for(auto b = 0u; b < batch_size; ++b) {
		auto input_image =
			neu::load_rgb_image_as_3ch_image_vector("../../../data/lena.bmp");
		auto cpu_input = std::get<0>(input_image);
		input.insert(input.end(), cpu_input.begin(), cpu_input.end());
	}

	neu::average_pooling_layer_parameter params;
	params
	.input_width(512)
	.filter_width(25)
	.input_channel_num(3)
	.stride(3)
	.pad(25/2)
	.batch_size(1)
	;
	auto pool = neu::make_uniform_average_pooling_layer(params);
	pool.forward(input);
	auto next_input = pool.get_next_input();
	neu::save_image_vector_as_images(neu::to_cpu_vector(next_input),
		params.output_width(), params.output_channel_num(), params.batch_size(),
		"output.bmp");
	//pool.backward(next_input);
	pool.backward(neu::to_gpu_vector(neu::cpu_vector(next_input.size(), 1)));
	auto prev_delta = pool.get_prev_delta();
	neu::save_image_vector_as_images(neu::to_cpu_vector(prev_delta),
		params.input_width(), params.input_channel_num(), params.batch_size(),
		"prev_delta.bmp");
}
