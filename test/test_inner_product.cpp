#define BOOST_TEST_MODULE TestInnerProduct
#include <boost/test/unit_test.hpp>
#include <random>

#include <neu/layer/inner_product.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(forward) {
	/*
	neu::cpu_vector weight = {
		1.f, 1.f, 1.f,
		1.f, 0.f, 0.f
	};
	*/
	neu::cpu_vector weight = {
		1.f, 1.f,
		1.f, 0.f,
		1.f, 0.f
	};
	neu::layer::inner_product ip(3, 2, 3, weight,
		neu::optimizer::fixed_learning_rate(0.001), queue);
	neu::gpu_vector input({
		0.f, 1.f, 0.f,
		1.f, 0.f, 1.f,
		0.f, 1.f, 0.f
	}, queue);
	neu::gpu_vector output(neu::layer::whole_output_size(ip), context);
	neu::layer::forward(ip, input, output, queue);
	queue.finish();
	BOOST_CHECK(output.size() == 6);
	CHECK_RANGE_EQUAL(neu::scalar, 6, output, (
		1.f, 0.f,
		2.f, 1.f,
		1.f, 0.f
	));
}
BOOST_AUTO_TEST_CASE(backward) {
	/*
	neu::cpu_vector weight = {
		1.f, 1.f, 1.f,
		1.f, 0.f, 0.f
	};
	*/
	neu::cpu_vector weight = {
		1.f, 1.f,
		1.f, 0.f,
		1.f, 0.f
	};
	neu::layer::inner_product ip(3, 2, 3, weight,
		neu::optimizer::fixed_learning_rate(0.001), queue);
	neu::gpu_vector delta({
		1.f, 0.f,
		2.f, 1.f,
		1.f, 0.f
	}, queue);
	neu::gpu_vector prev_delta(neu::layer::whole_input_size(ip), context);
	neu::layer::backward(ip, delta, prev_delta, queue);
	queue.finish();
	BOOST_CHECK(prev_delta.size() == 9);
	CHECK_RANGE_EQUAL(neu::scalar, 9, prev_delta, (
		1.f, 1.f, 1.f,
		3.f, 2.f, 2.f,
		1.f, 1.f, 1.f
	));
}
BOOST_AUTO_TEST_CASE(gradient_check_single) {
	const auto opt = neu::optimizer::fixed_learning_rate(1.0e-4f);

	neu::gpu_vector input({
		0.f, 1.f, 0.f,
		1.f, 0.f, 1.f,
		0.f, 1.f, 0.f
	}, queue);

	neu::gpu_vector teach({
		1.f, 0.f,
		0.f, 1.f,
		1.f, 0.f
	}, queue);

	neu::cpu_vector weight(6);
	std::random_device rd; std::mt19937 rand(rd());
	std::generate(weight.begin(), weight.end(),
		[&rand, dist=std::normal_distribution<>(0.f, 1.f)]
		() mutable { return dist(rand); });

	neu::layer::inner_product ip(3, 2, 3, weight, opt, queue);
	neu::gpu_vector output(neu::layer::whole_output_size(ip), context);
	neu::layer::forward(ip, input, output, queue);
	neu::gpu_vector error(output.size(), context);
	neu::range::calc_last_layer_delta(output, teach, error, queue);
	neu::layer::backward_top(ip, error, queue);
	neu::layer::update(ip, queue);

	for(int i = 0; i < static_cast<int>(weight.size()); ++i) {
		const double numeric_grad = ip.del_weight(queue)[i];

		const auto theta = weight[i];
		const double eps = 1.0e-2f;

		const auto analytic_grad =  neu::calc_analytic_gradient(
			[this, i, weight, &opt, &input, &teach](neu::scalar theta_dash) mutable {
				weight[i] = theta_dash;
				neu::layer::inner_product ip(3, 2, 3, weight, opt, queue);
				neu::gpu_vector output(neu::layer::whole_output_size(ip), context);
				neu::layer::forward(ip, input, output, queue);
				return neu::range::half_square_error_sum(output, teach, queue)/3.;
			}, theta, eps);
		const auto relative_error = neu::calc_relative_error(numeric_grad, analytic_grad);
		BOOST_CHECK(relative_error < 1.0e-2f);
	}
}
BOOST_AUTO_TEST_CASE(gradient_check_multi) {
	const auto opt = neu::optimizer::fixed_learning_rate(1.0e-4f);
	const int batch_size = 10;
	const int input_dim = 5;
	const int output1_dim = 10;
	const int output2_dim = 6;

	auto random_vector_gen = [](int size) {
		neu::cpu_vector vec(size);
		std::random_device rd; std::mt19937 rand(rd());
		std::generate(vec.begin(), vec.end(),
			[&rand, dist=std::uniform_real_distribution<>(0.f, 1.f)]
			() mutable { return dist(rand); });
		return vec;
	};
	for(int t = 0; t < 100; ++t) {
		const auto cpu_input = random_vector_gen(input_dim*batch_size);
		const neu::gpu_vector input(cpu_input.begin(), cpu_input.end(), queue);

		const auto cpu_teach = random_vector_gen(output2_dim*batch_size);
		const neu::gpu_vector teach(cpu_teach.begin(), cpu_teach.end(), queue);

		const auto weight1 = random_vector_gen(input_dim*output1_dim);
		const auto weight2 = random_vector_gen(output1_dim*output2_dim);

		neu::layer::inner_product ip1(
			input_dim, output1_dim, batch_size, weight1, opt, queue);
		neu::layer::inner_product ip2(
			output1_dim, output2_dim, batch_size, weight2, opt, queue);
		neu::gpu_vector output1(neu::layer::whole_output_size(ip1), context);
		neu::gpu_vector output2(neu::layer::whole_output_size(ip2), context);
		neu::layer::forward(ip1, input, output1, queue);
		neu::layer::forward(ip2, output1, output2, queue);

		neu::gpu_vector delta2(output2.size(), context);
		neu::range::calc_last_layer_delta(output2, teach, delta2, queue);

		neu::gpu_vector delta1(output1.size(), context);
		neu::layer::backward(ip2, delta2, delta1, queue);
		neu::layer::backward_top(ip1, delta1, queue);

		neu::layer::update(ip1, queue);
		neu::layer::update(ip2, queue);

		for(int i = 0; i < static_cast<int>(weight1.size()); ++i) {
			const double numeric_grad = ip1.del_weight(queue)[i];

			const auto theta = weight1[i];
			const double eps = 1.0e-2f;

			const double analytic_grad = neu::calc_analytic_gradient(
				[this, i, batch_size, &weight1, &weight2, &opt, &input, &teach]
				(neu::scalar theta_dash) {
					auto weight_temp = weight1;
					weight_temp[i] = theta_dash;
					neu::layer::inner_product ip1(
						input_dim, output1_dim, batch_size, weight_temp, opt, queue);
					neu::layer::inner_product ip2(
						output1_dim, output2_dim, batch_size, weight2, opt, queue);
					neu::gpu_vector output1(neu::layer::whole_output_size(ip1), context);
					neu::gpu_vector output2(neu::layer::whole_output_size(ip2), context);
					neu::layer::forward(ip1, input, output1, queue);
					neu::layer::forward(ip2, output1, output2, queue);
					return neu::range::half_square_error_sum(output2, teach, queue)/batch_size;
				}, theta, eps);
			const auto relative_error = neu::calc_relative_error(numeric_grad, analytic_grad);
			//std::cout << numeric_grad << " " << analytic_grad << std::endl;
			BOOST_CHECK(relative_error < 1.0e-2f);
		}
	}
}
BOOST_AUTO_TEST_SUITE_END()
