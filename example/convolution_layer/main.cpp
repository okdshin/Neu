#include <iostream>
#include <neu/convolution_layer.hpp>
#include <neu/learning_rate_gen/weight_decay_and_momentum.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>
#include <neu/layers_algorithm.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	//std::random_device engine; std::mt19937 rand(engine());
	std::mt19937 rand(0);
	auto batch_size = 1u;
	neu::gpu_vector input;
	for(auto b = 0u; b < batch_size; ++b) {
		auto input_image =
			neu::load_rgb_image_as_3ch_image_vector("../../../data/lena256.bmp");
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
	auto output_channel_num = 1u;
	auto filter_width = 20u;
	auto stride = 2u;
	auto pad = filter_width/2u;

	auto conv_param = neu::convolution_layer_parameter()
		.input_width(input_width)
		.input_channel_num(input_channel_num)
		.batch_size(batch_size)
		.filter_width(filter_width)
		.stride(stride)
		.pad(pad)
		.output_channel_num(output_channel_num);
	auto output_width = conv_param.output_width();
	std::cout << "output_width: " << output_width << std::endl;

	auto g = [&rand, dist=std::uniform_real_distribution<>(-1.f, 1.f)]
		() mutable { return dist(rand); };

	neu::scalar base_lr = 0.01;
	neu::scalar momentum = 0.f;//0.9f;
	neu::scalar weight_decay = 0.;//0.004f;
	auto conv = neu::make_convolution_layer(conv_param, g, g,
		neu::make_weight_decay_and_momentum(base_lr, momentum, weight_decay,
			conv_param.weight_dim(), conv_param.bias_dim())
	);

	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream output_log("output.txt");
	neu::gpu_vector output(neu::layer_output_size(conv));
	neu::gpu_vector normalized_output(neu::layer_output_size(conv));
	neu::gpu_vector prev_delta(neu::layer_input_size(conv));
	{
		neu::cpu_vector cpu_filters(conv.get_filters().size());
		neu::range_copy(conv.get_filters(), cpu_filters);
		neu::save_image_vector_as_images(cpu_filters,
			filter_width, input_channel_num, output_channel_num,
			"0filter.bmp", 255.f);
	}
	for(auto i = 0u; i < 1000u; ++i) {
		conv.forward(input, output);
		neu::gpu_vector error(output.size());
		neu::cpu_vector cpu_output(neu::layer_output_size(conv));
		neu::range_copy(output, cpu_output);
		neu::cpu_vector normalized_cpu_output(cpu_output.size());
		neu::normalize(cpu_output.begin(), cpu_output.end(),
			normalized_cpu_output.begin());
		neu::range_copy(normalized_cpu_output, normalized_output);
		{
			auto mse = neu::mean_square_error(normalized_output, teach);
			mse_error_log << i << " " << mse << std::endl;
		}
		neu::calc_last_layer_delta(normalized_output, teach, error);
		conv.backward(error, prev_delta);
		conv.update();
		if(i%100 == 0) {
			neu::save_image_vector_as_images(neu::to_cpu_vector(conv.get_filters()),
				filter_width, input_channel_num, output_channel_num,
				"filter"+std::to_string(i)+".bmp", 255.f);

			neu::save_image_vector_as_images(neu::to_cpu_vector(conv.get_del_filters()),
				filter_width, input_channel_num, output_channel_num,
				"delta_filter"+std::to_string(i)+".bmp", 255.f);

			neu::save_image_vector_as_images(neu::to_cpu_vector(output),
				output_width, output_channel_num, batch_size,
				"output"+std::to_string(i)+".bmp", 255.f);
		}
	}
	boost::compute::system::finish();

}
