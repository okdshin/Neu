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
		auto input_image = neu::load_rgb_image_as_3ch_image_vector("../../../data/lena512.bmp");
		auto cpu_input = std::get<0>(input_image);
		input.insert(input.end(), cpu_input.begin(), cpu_input.end());
	}

	neu::max_pooling_layer_parameter params;
	params
	.input_width(512)
	.filter_width(7)
	.input_channel_num(3)
	.stride(2)
	.pad(3)
	.batch_size(batch_size)
	;
	auto pool = neu::make_max_pooling_layer(params);
	neu::gpu_vector output(neu::layer_output_dim(pool)*neu::layer_batch_size(pool));
	neu::layer_forward(pool, input, output);
	neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(output),
		neu::layer_output_width(pool), "output.bmp", 255);

	neu::gpu_vector prev_delta(neu::layer_input_dim(pool)*neu::layer_batch_size(pool));
	neu::layer_backward(pool, neu::gpu_vector(output.size(), 1.f), prev_delta);
	neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(prev_delta),
		neu::layer_input_width(pool), "prev_delta.bmp", 255);
}
