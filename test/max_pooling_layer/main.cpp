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
	//auto output_width = input_width/stride;

	neu::max_pooling_layer_parameter params;
	params
	.input_width(512)
	.filter_width(3)
	.input_channel_num(3)
	.stride(2)
	.pad(1)
	.batch_size(1)
	;
	std::cout << params.output_width() << std::endl;
	auto pool = neu::make_max_pooling_layer(params);
	std::cout << neu::layer_output_width(pool) << std::endl;
	std::cout << pool.output_width() << std::endl;
	neu::gpu_vector output(neu::layer_output_dim(pool)*neu::layer_batch_size(pool));
	neu::layer_forward(pool, input, output);
	neu::save_image_vector_as_images(neu::to_cpu_vector(output),
		neu::layer_output_width(pool), channel_num, batch_size, "output.bmp", 255);
	/*
	pool.backward(neu::cpu_vector(output.size(), 1));
	auto prev_delta = pool.get_prev_delta();
	neu::save_image_vector_as_images(neu::to_cpu_vector(prev_delta),
		input_width, channel_num, batch_size, "prev_delta.bmp");
	*/
}
