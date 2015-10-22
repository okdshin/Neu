#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/convolution_layer.hpp>
//#include <neu/learning_rate_gen/fixed_learning_rate_gen.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	//std::random_device rand{};
	std::mt19937 rand(0);
	auto batch_size = 1u;
	neu::gpu_vector input;
	for(auto b = 0u; b < batch_size; ++b) {
		auto input_image = neu::load_rgb_image_as_3ch_image_vector("../../../data/lena256.bmp");
		auto cpu_input = std::get<0>(input_image);
		input.insert(input.end(), cpu_input.begin(), cpu_input.end());
	}
	neu::gpu_vector teach;
	{
		auto teach_image = neu::load_rgb_image_as_3ch_image_vector("teach.bmp");
		auto cpu_teach = std::get<0>(teach_image);
		teach.insert(teach.end(), cpu_teach.begin(), cpu_teach.end());
	}
	auto input_width = 256u;
	auto input_channel_num = 3u;
	auto output_channel_num = 3u;
	auto filter_width = 11u;
	auto stride = 5u;
	auto pad = filter_width/2;
	auto output_width = (input_width-filter_width+1+2*pad)/stride;

	auto g = [&rand, dist=std::uniform_real_distribution<>(-1.f, 1.f)]
		() mutable { return dist(rand); };

	auto initial_filters = neu::make_random_gpu_vector(
		filter_width*filter_width*input_channel_num*output_channel_num, g);
	auto initial_bias = neu::make_random_gpu_vector(
		filter_width*filter_width*output_channel_num, g);
	neu::scalar base_lr = 0.1;
	neu::scalar momentum = 0.;//0.9;
	neu::scalar weight_decay = 0.;//0.004;
	auto conv = neu::make_convolution_layer(
		input_width, output_width, filter_width,
		input_channel_num, output_channel_num,
		stride, pad, batch_size,
		initial_filters, initial_bias,
		neu::weight_decay_and_momentum(base_lr, momentum, weight_decay,
			filter_width*filter_width*input_channel_num*output_channel_num,
			filter_width*filter_width*output_channel_num)
	);

	for(auto i = 0u; i < 10000u; ++i) {
		auto filters = conv.get_filters();
		conv.forward(input);
		auto next_input = conv.get_next_input();
		auto cpu_next_input = neu::to_cpu_vector(next_input);
		neu::cpu_vector normalized_cpu_next_input(cpu_next_input.size());
		neu::normalize(cpu_next_input.begin(), cpu_next_input.end(),
			normalized_cpu_next_input.begin());
		auto normalized_next_input = neu::to_gpu_vector(normalized_cpu_next_input);
		neu::gpu_vector error(next_input.size());
		boost::compute::transform(
			normalized_next_input.begin(), normalized_next_input.end(),
			teach.begin(), error.begin(), boost::compute::minus<neu::scalar>());
		conv.backward(error);
		//auto prev_delta = conv.get_prev_delta();
		conv.update();
		auto delta_filters = conv.get_del_filters();
		if(i%100 == 0) {
			neu::save_image_vector_as_images(neu::to_cpu_vector(filters),
				filter_width, input_channel_num, output_channel_num,
				"filter"+std::to_string(i)+".bmp");
			neu::save_image_vector_as_images(neu::to_cpu_vector(delta_filters),
				filter_width, input_channel_num, output_channel_num,
				"delta_filter"+std::to_string(i)+".bmp");
			neu::save_image_vector_as_images(neu::to_cpu_vector(next_input),
				output_width, output_channel_num, batch_size,
				"output"+std::to_string(i)+".bmp");
		}
		/*
		neu::save_image_vector_as_images(neu::to_cpu_vector(prev_delta),
			input_width, input_channel_num, batch_size,
			"prev_delta"+std::to_string(i)+".bmp");
		*/
	}
}
