#include <iostream>
#include <neu/layer/convolution.hpp>
#include <neu/optimizer/momentum.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>
#include <neu/range/algorithm.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	auto& queue = boost::compute::system::default_queue();
	auto context = boost::compute::system::default_context();

	//std::random_device engine; std::mt19937 rand(engine());
	std::mt19937 rand(0);
	auto batch_size = 1;
	neu::gpu_vector input;
	for(auto b = 0; b < batch_size; ++b) {
		auto input_image =
			neu::load_rgb_image_as_3ch_image_vector("../../../data/lena256.bmp");
		auto cpu_input = std::get<0>(input_image);
		input.insert(input.end(), cpu_input.begin(), cpu_input.end());
	}
	neu::gpu_vector teach;
	int teach_width;
	{
		auto teach_image =
			neu::load_rgb_image_as_3ch_image_vector("../../../data/teach.bmp");
		auto cpu_teach = std::get<0>(teach_image);
		teach_width = std::get<1>(teach_image);
		teach.insert(teach.end(), cpu_teach.begin(), cpu_teach.end());
	}

	constexpr neu::scalar base_lr = 0.1;
	constexpr neu::scalar momentum = 0.0;

	auto input_width = 256;
	auto input_channel_num = 3;
	auto output_channel_num = 1;
	auto filter_width = 13;
	auto stride = 2;
	auto pad = filter_width/2;

	auto g = [&rand, dist=std::uniform_real_distribution<>(-1.f, 1.f)]
		() mutable { return dist(rand); };

	neu::layer::geometric_layer_property glp{
		input_width, filter_width, input_channel_num, output_channel_num, stride, pad};
	auto output_width = neu::layer::output_width(glp);
	if(output_width != teach_width) {
		std::cout << "output image width(=" << output_width << ") is not equal to teach image one(=" << teach_width << ") (maybe filter width and/or pad are wrong)." << std::endl;
		return 0;
	}
	auto conv = make_convolution(glp, batch_size, g,
		neu::optimizer::momentum(base_lr, momentum,
			neu::layer::filters_size(glp), queue), queue);

	std::ofstream mse_error_log("mse_error.txt");
	std::ofstream output_log("output.txt");

	neu::gpu_vector output(neu::layer::whole_output_size(conv));
	neu::gpu_vector normalized_output(output.size());
	neu::gpu_vector error(normalized_output.size(), context);
	neu::gpu_vector prev_delta(neu::layer::whole_input_size(conv));

	neu::save_image_vector_as_images(conv.filters(queue),
		filter_width, input_channel_num, output_channel_num,
		[](std::size_t i, std::size_t k){
			return "0filter"+std::to_string(i)+"_"+std::to_string(k)+".bmp"; }, 255.f);

	for(auto i = 0; i < 2000; ++i) {
		neu::layer::forward(conv, input, output, queue);

		{
			auto cpu_output = neu::to_cpu_vector(output, queue);
			neu::cpu_vector normalized_cpu_output(cpu_output.size());
			neu::normalize(cpu_output.begin(), cpu_output.end(),
				normalized_cpu_output.begin());
			neu::range::copy(normalized_cpu_output, normalized_output, queue);
		}
		{
			auto mse = neu::range::mean_square_error(normalized_output, teach, queue);
			mse_error_log << i << " " << mse << std::endl;
		}
		neu::range::calc_last_layer_delta(normalized_output, teach, error, queue);
		neu::layer::backward(conv, error, prev_delta, queue);
		neu::layer::update(conv, queue);

		if(i%100 == 0) {
			neu::save_image_vector_as_images(conv.filters(queue),
				filter_width, input_channel_num, output_channel_num,
				[i](int b, int k) {
					return "filter"+std::to_string(i)+".bmp";
				}, 
				255.f);

			neu::save_image_vector_as_images(conv.del_filters(queue),
				filter_width, input_channel_num, output_channel_num,
				[i](int b, int k) {
					return "delta"+std::to_string(i)+".bmp";
				}, 255.f);

			neu::save_image_vector_as_images(neu::to_cpu_vector(output, queue),
				output_width, output_channel_num, batch_size,
				[i](int b, int k) {
					return "output"+std::to_string(i)+".bmp";
				}, 255.f);
		}
	}
	boost::compute::system::finish();

}
