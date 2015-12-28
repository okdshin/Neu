#include <iostream>
#define NEU_DISABLE_ASSERT_FOR_HEAVY_CALCULATION
#include <neu/layer/local_contrast_normalization.hpp>
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

	neu::layer::geometric_layer_property glp{512, 11, 3, 3, 1, 5};
	auto lcn = neu::layer::local_contrast_normalization(
		glp, batch_size, 0.001, 0.5, context);

	neu::gpu_vector output(neu::layer::whole_output_size(lcn));
	neu::layer::forward(lcn, input, output, queue);
	neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(output, queue),
		neu::layer::output_width(lcn), "output.bmp", 255);
	
	neu::gpu_vector prev_delta(neu::layer::whole_input_size(lcn));
	neu::layer::backward(lcn,
		neu::gpu_vector(output.size(), 1.f, queue), prev_delta, queue);
	neu::save_3ch_image_vector_as_rgb_image(neu::to_cpu_vector(prev_delta, queue),
		neu::layer::input_width(lcn), "prev_delta.bmp", 255);

}
