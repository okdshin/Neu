#define BOOST_TEST_MODULE TestConvolution
#include <boost/test/unit_test.hpp>

#include <neu/layer/convolution.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>

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
	neu::layer::convolution conv(glp, 1, filters,
		neu::optimizer::fixed_learning_rate(0.001), queue);
	neu::gpu_vector input({
		0.f, 1.f, 0.f,
		1.f, 0.f, 1.f,
		0.f, 1.f, 0.f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		0.f, 1.f, 0.f
	}, queue);
	neu::gpu_vector output(neu::layer::whole_output_size(conv), context);
	neu::layer::forward(conv, input, output, queue);
	queue.finish();
	BOOST_CHECK(output.size() == 27);
	CHECK_RANGE_EQUAL(neu::scalar, 27, output, (
		1.f, 2.5f, 1.f,
		2.5f, 2.f, 1.5f,
		1.f, 1.5f, 1.f,

		0.2f, 0.3f, 0.2f,
		0.3f, 1.4f, 0.3f,
		0.2f, 0.3f, 0.2f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		0.f, 1.f, 0.f
	));
}

/*
BOOST_AUTO_TEST_CASE(backward) {
	neu::cpu_vector weight = {
		1.f, 1.f, 1.f,
		1.f, 0.f, 0.f
	};
	neu::layer::inner_product ip(3, 2, 3, weight,
		neu::optimizer::fixed_learning_rate(0.001), queue);
	neu::gpu_vector next_delta({
		1.f, 0.f,
		2.f, 1.f,
		1.f, 0.f
	}, queue);
	neu::gpu_vector delta(neu::layer::whole_input_size(ip), context);
	neu::layer::backward(ip, next_delta, delta, queue);
	queue.finish();
	BOOST_CHECK(delta.size() == 9);
	CHECK_RANGE_EQUAL(neu::scalar, 9, delta, (
		1.f, 1.f, 1.f,
		3.f, 2.f, 2.f,
		1.f, 1.f, 1.f
	));
}
*/
/*
BOOST_AUTO_TEST_CASE(gradient_check_single) {
	neu::layer::geometric_layer_property glp{3, 3, 2, 3, 1, 1};
	const auto opt = neu::optimizer::fixed_learning_rate(1.0e-4f);

	neu::gpu_vector input({
		0.f, 1.f, 0.f,
		1.f, 0.f, 1.f,
		0.f, 1.f, 0.f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		0.f, 1.f, 0.f
	}, queue);

	neu::gpu_vector teach({
		1.f, 2.5f, 1.f,
		2.5f, 2.f, 1.5f,
		1.f, 1.5f, 1.f,

		0.2f, 0.3f, 0.2f,
		0.3f, 1.4f, 0.3f,
		0.2f, 0.3f, 0.2f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		0.f, 1.f, 0.f
	}, queue);

	for(int t = 0; t < 100; ++t) {
		neu::cpu_vector weight(54);
		std::random_device rd; std::mt19937 rand(rd());
		std::generate(weight.begin(), weight.end(),
			[&rand, dist=std::uniform_real_distribution<>(0.f, 1.f)]
			() mutable { return dist(rand); });

		neu::layer::convolution conv(glp, 1, weight, opt, queue);
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

			const auto relative_error =  neu::check_gradient(
				[this, i, &weight, glp, &opt, &input, &teach]
				(neu::scalar theta_dash) {
					auto weight_temp = weight;
					weight_temp[i] = theta_dash;
					neu::layer::convolution conv(glp, 1, weight_temp, opt, queue);
					neu::gpu_vector output(neu::layer::whole_output_size(conv), context);
					neu::layer::forward(conv, input, output, queue);
					return neu::range::half_square_error_sum(output, teach, queue);
				}, theta, eps, numeric_grad);
			BOOST_CHECK(relative_error < 1.0e-2f);
		}
	}
}
*/

BOOST_AUTO_TEST_CASE(gradient_check_multi) {
	neu::layer::geometric_layer_property glp1{3, 3, 2, 3, 1, 1};
	neu::layer::geometric_layer_property glp2{3, 3, 3, 3, 1, 1};
	const auto opt = neu::optimizer::fixed_learning_rate(1.0e-4f);

	neu::gpu_vector input({
		0.f, 1.f, 0.f,
		1.f, 0.f, 1.f,
		0.f, 1.f, 0.f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		0.f, 1.f, 0.f
	}, queue);

	neu::gpu_vector teach({
		1.f, 2.5f, 1.f,
		2.5f, 2.f, 1.5f,
		1.f, 1.5f, 1.f,

		0.2f, 0.3f, 0.2f,
		0.3f, 1.4f, 0.3f,
		0.2f, 0.3f, 0.2f,

		1.f, 0.f, 0.f,
		0.f, 0.f, 1.f,
		0.f, 1.f, 0.f
	}, queue);

	for(int t = 0; t < 100; ++t) {
		auto weight_gen = [](int size) {
			neu::cpu_vector weight(size);
			std::random_device rd; std::mt19937 rand(rd());
			std::generate(weight.begin(), weight.end(),
				[&rand, dist=std::uniform_real_distribution<>(0.f, 1.f)]
				() mutable { return dist(rand); });
			return weight;
		};
		auto weight1 = weight_gen(54);
		auto weight2 = weight_gen(81);

		neu::layer::convolution conv1(glp1, 1, weight1, opt, queue);
		neu::layer::convolution conv2(glp2, 1, weight2, opt, queue);
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
			const double eps = 1.0e-2f;

			const double analytic_grad = neu::calc_analytic_gradient(
				[this, i, &weight1, &weight2, &glp1, &glp2, &opt, &input, &teach]
				(neu::scalar theta_dash) {
					auto weight_temp = weight1;
					weight_temp[i] = theta_dash;
					neu::layer::convolution conv1(glp1, 1, weight_temp, opt, queue);
					neu::layer::convolution conv2(glp2, 1, weight2, opt, queue);
					neu::gpu_vector output1(neu::layer::whole_output_size(conv1), context);
					neu::gpu_vector output2(neu::layer::whole_output_size(conv2), context);
					neu::layer::forward(conv1, input, output1, queue);
					neu::layer::forward(conv2, output1, output2, queue);
					return neu::range::half_square_error_sum(output2, teach, queue);
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
	}
}

BOOST_AUTO_TEST_SUITE_END()
