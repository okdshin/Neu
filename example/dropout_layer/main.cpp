#include <iostream>
#include <neu/layer/io.hpp>
#include <neu/layer/dropout.hpp>
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

	auto input_width = 512u;
	auto output_width = 512u;

	auto dropout = neu::layer::dropout(input_width*input_width*3, batch_size, 0.5, queue);

	for(auto i = 0; i < 3; ++i) {
		neu::gpu_vector output(neu::layer::whole_output_size(dropout), context);
		neu::layer::forward(dropout, input, output, queue);
		neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(output, queue),
			output_width, std::to_string(i)+"output.bmp", 255);
		
		neu::gpu_vector prev_delta(neu::layer::whole_input_size(dropout), context);
		neu::layer::backward(dropout,
			neu::gpu_vector(output.size(), 1.f, queue), prev_delta, queue);
		neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(prev_delta, queue),
			input_width, std::to_string(i)+"prev_delta.bmp", 255);
		neu::layer::output_to_file(dropout, "layer.yaml", queue);

		dropout.update(queue);
	}

	/*
	neu::layer::dropout d(input_width*input_width*3, batch_size, 0.5, queue);
	neu::layer::any_layer dropout2 = d;//std::ref(d);
	dropout2.update(queue);
	*/

	/*
	neu::layer::output_to_file(dropout, "layer.yaml", queue);
	auto loaded = neu::layer::input_from_file("layer.yaml", queue);
	neu::layer::output_to_file(loaded, "loaded_layer.yaml", queue);
	loaded.update(queue);
	*/

	{
		auto loaded = neu::layer::input_from_file("layer.yaml", queue);
		neu::gpu_vector output(neu::layer::whole_output_size(loaded), context);
		auto output_range = neu::range::to_range(output);
		loaded.forward(neu::range::to_range(input), output_range, queue);
		neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(output, queue),
			output_width, "output_loaded.bmp", 255);
		
		neu::gpu_vector prev_delta(neu::layer::whole_input_size(loaded), context);
		auto prev_delta_range = neu::range::to_range(prev_delta);
		neu::layer::backward(loaded,
			neu::range::to_range(neu::gpu_vector(output.size(), 1.f, queue)),
			prev_delta_range, queue);
		neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(prev_delta, queue),
			input_width, "prev_delta_loaded.bmp", 255);
	}
	queue.finish();

}
