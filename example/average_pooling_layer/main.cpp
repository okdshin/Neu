#include <iostream>
#include <neu/layer/average_pooling.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	auto& queue = boost::compute::system::default_queue();
	auto context = boost::compute::system::default_context();

	auto batch_size = 1u;
	neu::gpu_vector input;
	for(auto b = 0u; b < batch_size; ++b) {
		auto input_image =
			neu::load_rgb_image_as_3ch_image_vector("../../../data/lena512.bmp");
		auto cpu_input = std::get<0>(input_image);
		input.insert(input.end(), cpu_input.begin(), cpu_input.end());
	}

	neu::layer::geometric_layer_property glp{512, 20, 3, 3, 2, 10};
	auto pool = neu::layer::make_uniform_average_pooling(glp, batch_size, queue);

	neu::gpu_vector output(neu::layer::whole_output_size(pool));
	neu::layer::forward(pool, input, output, queue);
	neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(output, queue),
		neu::layer::output_width(pool), "output.bmp", 255);
	
	neu::gpu_vector prev_delta(neu::layer::whole_input_size(pool));
	neu::layer::backward(pool, neu::gpu_vector(output.size(), 1.f), prev_delta, queue);
	neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(prev_delta, queue),
		neu::layer::input_width(pool), "prev_delta.bmp", 255);

	/*
	neu::average_pooling_layer_parameter params;
	params
	.input_width(512)
	.filter_width(7)
	.input_channel_num(3)
	.stride(2)
	.pad(3)
	.batch_size(batch_size)
	;
	auto pool = neu::make_uniform_average_pooling_layer(params);
	neu::gpu_vector output(neu::layer_output_dim(pool)*neu::layer_batch_size(pool));
	neu::layer_forward(pool, input, output);
	neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(output),
		neu::layer_output_width(pool), "output.bmp", 255);

	neu::gpu_vector prev_delta(neu::layer_input_dim(pool)*neu::layer_batch_size(pool));
	neu::layer_backward(pool, neu::gpu_vector(output.size(), 1.f), prev_delta);
	neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(prev_delta),
		neu::layer_input_width(pool), "prev_delta.bmp", 255);
	*/
}
