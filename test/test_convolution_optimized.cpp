#define BOOST_TEST_MODULE TestConvolutionOptimized
#include <boost/test/unit_test.hpp>
#include <boost/progress.hpp>

#include <neu/layer/convolution_optimized.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>
#include <neu/vector_io.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(forward) {
	neu::layer::geometric_layer_property glp{3, 3, 2, 3, 1, 1};
	neu::cpu_vector filters = {
		0.5f, 0.5f, 0.5f,
		0.5f, 0.5f, 0.5f,
		0.5f, 0.5f, 0.5f,

		0.f, 0.f, 0.f,
		0.f, 0.f, 0.f,
		0.f, 0.f, 1.f,


		0.1f, 0.1f, 0.1f,
		0.1f, 0.1f, 0.1f,
		0.1f, 0.1f, 0.1f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 0.f,
		0.f, 0.f, 0.f,


		0.f, 0.f, 0.f,
		0.f, 0.f, 0.f,
		0.f, 0.f, 0.f,

		0.f, 0.f, 0.f,
		0.f, 1.f, 0.f,
		0.f, 0.f, 0.f
	};
	neu::layer::convolution_optimized conv(glp, 2, filters,
		neu::optimizer::fixed_learning_rate(0.001), queue);
	neu::gpu_vector input({
		0.f, 1.f, 0.f,
		1.f, 0.f, 1.f,
		0.f, 1.f, 0.f,
		//0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		0.f, 1.f, 0.f,


		1.f, 1.f, 1.f,
		1.f, 1.f, 1.f,
		1.f, 1.f, 1.f,

		1.f, 1.f, 0.f,
		1.f, 0.f, 1.f,
		0.f, 1.f, 1.f
	}, queue);
	neu::gpu_vector output(neu::layer::whole_output_size(conv), context);
	neu::layer::forward(conv, input, output, queue);
	queue.finish();
	auto reordered_input = conv.reordered_input(queue);
	neu::print(std::cout, reordered_input, 9);
	neu::print(std::cout, output, 54);
	BOOST_CHECK(output.size() == 54);
	CHECK_RANGE_EQUAL(neu::scalar, 54, output, (
		1.f, 2.5f, 1.f,
		2.5f, 2.f, 1.5f,
		1.f, 1.5f, 1.f,

		0.2f, 0.3f, 0.2f,
		0.3f, 1.4f, 0.3f,
		0.2f, 0.3f, 0.2f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		0.f, 1.f, 0.f,


		2.f, 4.f, 2.f,
		4.f, 5.5f, 3.f,
		2.f, 3.f, 2.f,

		0.4f, 0.6f, 0.4f,
		0.6f, 1.9f, 1.6f,
		0.4f, 1.6f, 0.4f,

		1.f, 1.f, 0.f,
		1.f, 0.f, 1.f,
		0.f, 1.f, 1.f
	));
}

BOOST_AUTO_TEST_CASE(gradient_check_single) {
	neu::layer::geometric_layer_property glp{10, 3, 2, 3, 1, 1};
	const auto opt = neu::optimizer::fixed_learning_rate(1.0e-4f);
	const int batch_size = 100;

	auto random_vector_gen = [](int size) {
		neu::cpu_vector vec(size);
		std::mt19937 rand(0);
		std::generate(vec.begin(), vec.end(),
			[&rand, dist=std::uniform_real_distribution<>(0.f, 1.f)]
			() mutable { return dist(rand); });
		return vec;
	};

	const auto t_limit = 1;
	boost::progress_display progress(t_limit);
	for(int t = 0; t < t_limit; ++t) {
		const auto cpu_input = random_vector_gen(neu::layer::input_dim(glp)*batch_size);
		const neu::gpu_vector input(cpu_input.begin(), cpu_input.end(), queue);

		const auto cpu_teach = random_vector_gen(neu::layer::output_dim(glp)*batch_size);
		const neu::gpu_vector teach(cpu_teach.begin(), cpu_teach.end(), queue);

		const auto weight = random_vector_gen(neu::layer::filters_size(glp));

		neu::layer::convolution_optimized conv(glp, batch_size, weight, opt, queue);
		neu::gpu_vector output(neu::layer::whole_output_size(conv), context);
		neu::layer::forward(conv, input, output, queue);
		neu::gpu_vector error(output.size(), context);
		neu::range::calc_last_layer_delta(output, teach, error, queue);
		neu::layer::backward_top(conv, error, queue);
		neu::layer::update(conv, queue);

		for(int i = 0; i < static_cast<int>(weight.size()); ++i) {
			const double numeric_grad = conv.del_filters(queue)[i];

			const auto theta = weight[i];
			const double eps = 1.0e-2f;

			const double analytic_grad = neu::calc_analytic_gradient(
				[this, i, batch_size, &weight, &glp, &opt, &input, &teach]
				(neu::scalar theta_dash) {
					auto weight_temp = weight;
					weight_temp[i] = theta_dash;
					neu::layer::convolution_optimized conv(
						glp, batch_size, weight_temp, opt, queue);
					neu::gpu_vector output(neu::layer::whole_output_size(conv), context);
					neu::layer::forward(conv, input, output, queue);
					return neu::range::half_square_error_sum(output, teach, queue)/batch_size;
				}, theta, eps);
			const auto relative_error =
				neu::calc_relative_error(analytic_grad, numeric_grad);
			/*
			std::cout << "numeric_grad:\t" << numeric_grad << std::endl;
			std::cout << "analytic_grad:\t" << analytic_grad << std::endl;
			std::cout << "relative_error:\t" << relative_error << std::endl;
			std::cout << "\n";
			*/
			BOOST_CHECK(relative_error < 1.0e-2f);
		}
		++progress;
	}
}
BOOST_AUTO_TEST_CASE(gradient_check_multi) {
	neu::layer::geometric_layer_property glp1{10, 3, 2, 3, 1, 1};
	neu::layer::geometric_layer_property glp2{10, 3, 3, 3, 1, 1};
	const auto opt = neu::optimizer::fixed_learning_rate(1.0e-4f);
	const int batch_size = 100;

	auto random_vector_gen = [](int size) {
		neu::cpu_vector vec(size);
		std::mt19937 rand(0);
		std::generate(vec.begin(), vec.end(),
			[&rand, dist=std::uniform_real_distribution<>(0.f, 1.f)]
			() mutable { return dist(rand); });
		return vec;
	};
	const auto t_limit = 100;
	boost::progress_display progress(t_limit);
	for(int t = 0; t < t_limit; ++t) {
		const auto cpu_input = random_vector_gen(neu::layer::input_dim(glp1)*batch_size);
		const neu::gpu_vector input(cpu_input.begin(), cpu_input.end(), queue);

		const auto cpu_teach = random_vector_gen(neu::layer::output_dim(glp2)*batch_size);
		const neu::gpu_vector teach(cpu_teach.begin(), cpu_teach.end(), queue);

		const auto weight1 = random_vector_gen(neu::layer::filters_size(glp1));
		const auto weight2 = random_vector_gen(neu::layer::filters_size(glp2));

		neu::layer::convolution_optimized conv1(glp1, batch_size, weight1, opt, queue);
		neu::layer::convolution_optimized conv2(glp2, batch_size, weight2, opt, queue);
		neu::gpu_vector output1(neu::layer::whole_output_size(conv1), context);
		neu::gpu_vector output2(neu::layer::whole_output_size(conv2), context);
		neu::layer::forward(conv1, input, output1, queue);
		neu::layer::forward(conv2, output1, output2, queue);

		neu::gpu_vector delta2(output2.size(), context);
		neu::range::calc_last_layer_delta(output2, teach, delta2, queue);

		neu::gpu_vector delta1(neu::layer::whole_output_size(conv1), context);
		neu::layer::backward(conv2, delta2, delta1, queue);
		neu::layer::backward_top(conv1, delta1, queue);

		neu::layer::update(conv1, queue);
		neu::layer::update(conv2, queue);

		for(int i = 0; i < static_cast<int>(weight1.size()); ++i) {
			const double numeric_grad = conv1.del_filters(queue)[i];

			const double theta = weight1[i];
			const double eps = 1.0e-1f;

			const double analytic_grad = neu::calc_analytic_gradient(
				[this, i, &weight1, &weight2, &glp1, &glp2, &opt, &input, &teach]
				(neu::scalar theta_dash) {
					auto weight_temp = weight1;
					weight_temp[i] = theta_dash;
					neu::layer::convolution_optimized conv1(
						glp1, batch_size, weight_temp, opt, queue);
					neu::layer::convolution_optimized conv2(
						glp2, batch_size, weight2, opt, queue);
					neu::gpu_vector output1(
						neu::layer::whole_output_size(conv1), context);
					neu::gpu_vector output2(
						neu::layer::whole_output_size(conv2), context);
					neu::layer::forward(conv1, input, output1, queue);
					neu::layer::forward(conv2, output1, output2, queue);
					return neu::range::half_square_error_sum(output2, teach, queue)/batch_size;
				}, theta, eps);
			const auto relative_error =
				neu::calc_relative_error(analytic_grad, numeric_grad);
			/*
			std::cout << "numeric_grad:\t" << numeric_grad << std::endl;
			std::cout << "analytic_grad:\t" << analytic_grad << std::endl;
			std::cout << "relative_error:\t" << relative_error << std::endl;
			std::cout << "\n";
			*/
			BOOST_CHECK(relative_error < 1.0e-2f);
		}
		++progress;
	}
}
BOOST_AUTO_TEST_SUITE_END()
