#define BOOST_TEST_MODULE TestMaxPooling
#include <boost/test/unit_test.hpp>

#include <neu/layer/inner_product.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(forward) {
	neu::cpu_vector weight = {
		1.f, 1.f, 1.f,
		1.f, 0.f, 0.f
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

BOOST_AUTO_TEST_CASE(gradient_check) {
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

	for(int t = 0; t < 10; ++t) {
		neu::cpu_vector weight(6);
		std::random_device rd; std::mt19937 rand(rd());
		std::generate(weight.begin(), weight.end(),
			[&rand, dist=std::normal_distribution<>(0.f, 1.f)]
			() mutable { return dist(rand); });

		for(int i = 0; i < weight.size(); ++i) {
			neu::layer::inner_product ip(3, 2, 3, weight, opt, queue);
			neu::gpu_vector output(neu::layer::whole_output_size(ip), context);
			neu::layer::forward(ip, input, output, queue);
			neu::gpu_vector error(output.size(), context);
			neu::range::calc_last_layer_delta(output, teach, error, queue);
			neu::layer::backward_top(ip, error, queue);
			neu::layer::update(ip, queue);
			const double numeric_grad = ip.del_weight(queue)[i];
			std::cout << "numeric_grad: " << numeric_grad << std::endl;

			const double eps = 1.0e-2f;
			/*
			const auto eps = weight[i] == 0 ? std::sqrt(std::numeric_limits<float>::epsilon())
				: std::sqrt(std::numeric_limits<float>::epsilon())*std::abs(weight[i]);
			*/
			std::cout << "eps: " << eps << std::endl;

			auto weight_plus = weight;
			weight_plus[i] += eps;
			neu::layer::inner_product ip_plus(3, 2, 3, weight_plus, opt, queue);
			neu::gpu_vector output_plus(neu::layer::whole_output_size(ip_plus), context);
			neu::layer::forward(ip_plus, input, output_plus, queue);
			const double mse_plus = neu::range::mean_square_error(output_plus, teach, queue);
			std::cout << "mse_plus: " << mse_plus << std::endl;

			auto weight_minus = weight;
			weight_minus[i] -= eps;
			neu::layer::inner_product ip_minus(3, 2, 3, weight_minus, opt, queue);
			neu::gpu_vector output_minus(neu::layer::whole_output_size(ip_minus), context);
			neu::layer::forward(ip_minus, input, output_minus, queue);
			const double mse_minus = neu::range::mean_square_error(output_minus, teach, queue);
			std::cout << "mse_minus: " << mse_minus << std::endl;

			const double analytic_grad = (mse_plus-mse_minus)/(2*eps);
			std::cout << "analytic_grad: " << analytic_grad << std::endl;

			queue.finish();

			if(numeric_grad != 0.f || analytic_grad != 0.f) {
				const auto relative_error = std::abs(analytic_grad-numeric_grad)
					/std::max(std::abs(analytic_grad), std::abs(numeric_grad));
					// /std::abs(weight[i]);
				std::cout << "relative_error: " << relative_error << std::endl;
				BOOST_CHECK(relative_error < 1.0e-2f);
			}
			std::cout << "\n";
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
