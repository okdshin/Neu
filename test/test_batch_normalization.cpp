#define BOOST_TEST_MODULE TestBatchNormalization
#include <boost/test/unit_test.hpp>

#include <neu/layer/batch_normalization.hpp>
#include <neu/layer/inner_product.hpp>
#include <neu/optimizer/fixed_learning_rate.hpp>
#include <random>

#include "check_tool.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(forward) {
	const auto batch_size = 100000u;
	const auto input_dim = 3;
	const auto mean_gt = 42;
	const auto variance_gt = 101;

	neu::layer::batch_normalization bn(batch_size, input_dim,
		neu::cpu_vector(input_dim, 1.f),
		neu::cpu_vector(input_dim, 0.f),
		neu::optimizer::fixed_learning_rate(0.01),
		neu::optimizer::fixed_learning_rate(0.01),
		queue);

	const auto cpu_input = [input_dim, batch_size, mean_gt, variance_gt]() {
		neu::cpu_vector cpu_input(input_dim*batch_size);
		std::mt19937 rng(0);
		std::generate(cpu_input.begin(), cpu_input.end(),
			[&rng, dist=std::normal_distribution<>(mean_gt, std::sqrt(variance_gt))]
				() mutable { return dist(rng); });
		return cpu_input;
	}();
	const auto input = neu::to_gpu_vector(cpu_input, queue);

	neu::gpu_vector output(neu::layer::whole_output_size(bn), context);
	neu::layer::forward(bn, input, output, queue);

	neu::gpu_vector output2(neu::layer::whole_output_size(bn), context);
	neu::layer::forward(bn, input, output2, queue);

	const auto cpu_output = neu::to_cpu_vector(output, queue);
	const auto cpu_output2 = neu::to_cpu_vector(output2, queue);

	for(int i = 0; i < static_cast<int>(cpu_output.size()); ++i) {
		BOOST_CHECK(std::abs(cpu_output[i]-cpu_output2[i]) < 1.0e-17f);
	}

	const auto mean = bn.mean(queue);
	BOOST_CHECK(mean.size() == input_dim);
	for(const auto m : mean) {
		BOOST_CHECK_CLOSE(m, mean_gt, 5.f);
	}

	const auto variance = bn.variance(queue);
	BOOST_CHECK(variance.size() == input_dim);
	for(const auto v : variance) {
		BOOST_CHECK_CLOSE(v, variance_gt, 5.f);
	}
}

BOOST_AUTO_TEST_CASE(backward) {
}

BOOST_AUTO_TEST_CASE(gradient_check_single) {
	const auto batch_size = 10000u;
	const auto input_dim = 3;
	const auto opt = neu::optimizer::fixed_learning_rate(1.0e-4f);

	const auto vecgen = [input_dim, batch_size](float mean, float variance) {
		neu::cpu_vector cpu_input(input_dim*batch_size);
		std::mt19937 rng(0);
		std::generate(cpu_input.begin(), cpu_input.end(),
			[&rng, dist=std::normal_distribution<>(mean, std::sqrt(variance))]
				() mutable { return dist(rng); });
		return cpu_input;
	};

	const auto input = neu::to_gpu_vector(vecgen(101, 42), queue);
	const auto teach = neu::to_gpu_vector(vecgen(10, 4), queue);

	const neu::cpu_vector gamma(input_dim, 1.f);
	const neu::cpu_vector beta(input_dim, 0.f);
	neu::layer::batch_normalization bn(
		batch_size, input_dim, gamma, beta, opt, opt, queue);

	neu::gpu_vector output(neu::layer::whole_output_size(bn), context);
	neu::layer::forward(bn, input, output, queue);
	neu::gpu_vector error(output.size(), context);
	neu::range::calc_last_layer_delta(output, teach, error, queue);
	neu::layer::backward_top(bn, error, queue);
	neu::layer::update(bn, queue);

	// gamma
	for(int i = 0; i < input_dim; ++i) {
		const double numeric_grad = bn.del_gamma(queue)[i];

		const auto theta = gamma[i];
		const double eps = 1.0e-2f;

		const double analytic_grad =  neu::calc_analytic_gradient(
			[this, i, batch_size, &gamma, &beta, &opt, &input, &teach]
			(neu::scalar theta_dash) mutable {
				auto temp_gamma = gamma;
				temp_gamma[i] = theta_dash;
				neu::layer::batch_normalization bn(
					batch_size, input_dim, temp_gamma, beta, opt, opt, queue);
				neu::gpu_vector output(neu::layer::whole_output_size(bn), context);
				neu::layer::forward(bn, input, output, queue);
				return neu::range::half_square_error_sum(output, teach, queue)/batch_size;
			}, theta, eps);
		const auto relative_error = neu::calc_relative_error(numeric_grad, analytic_grad);
		BOOST_CHECK(relative_error < 1.0e-2f);
	}

	// beta
	for(int i = 0; i < input_dim; ++i) {
		const double numeric_grad = bn.del_beta(queue)[i];

		const auto theta = beta[i];
		const double eps = 1.0e-2f;

		const double analytic_grad =  neu::calc_analytic_gradient(
			[this, i, batch_size, &gamma, &beta, &opt, &input, &teach]
			(neu::scalar theta_dash) {
				auto temp_beta = beta;
				temp_beta[i] = theta_dash;
				neu::layer::batch_normalization bn(
					batch_size, input_dim, gamma, temp_beta, opt, opt, queue);
				neu::gpu_vector output(neu::layer::whole_output_size(bn), context);
				neu::layer::forward(bn, input, output, queue);
				return neu::range::half_square_error_sum(output, teach, queue)/batch_size;
			}, theta, eps);
		const auto relative_error = neu::calc_relative_error(numeric_grad, analytic_grad);
		BOOST_CHECK(relative_error < 1.0e-2f);
	}
}

BOOST_AUTO_TEST_CASE(gradient_check_multi) {
	const auto opt = neu::optimizer::fixed_learning_rate(1.0e-4f);
	const auto batch_size = 1000u;
	const auto input_dim = 3;

	const auto vecgen = [input_dim, batch_size](float mean, float variance) {
		neu::cpu_vector vec(input_dim*batch_size);
		std::mt19937 rng(0);
		std::generate(vec.begin(), vec.end(),
			[&rng, dist=std::normal_distribution<>(mean, std::sqrt(variance))]
				() mutable { return dist(rng); });
		return vec;
	};

	std::mt19937 rng(0);
	auto rand = [&rng, dist=std::uniform_int_distribution<>(1, 100)]
		() mutable { return dist(rng); };

	for(int t = 0; t < 10; ++t) {
		const auto input = neu::to_gpu_vector(vecgen(rand(), rand()), queue);
		const auto teach = neu::to_gpu_vector(vecgen(rand(), rand()), queue);

		const auto weight = [&rng, input_dim]() {
			neu::cpu_vector vec(input_dim*input_dim);
			std::generate(vec.begin(), vec.end(),
				[&rng, dist=std::uniform_real_distribution<>(0.f, 1.f)]
				() mutable { return dist(rng); });
			return vec;
		}();
		neu::layer::inner_product ip(
			input_dim, input_dim, batch_size, weight, opt, queue);

		const neu::cpu_vector gamma(input_dim, 1.f);
		const neu::cpu_vector beta(input_dim, 0.f);
		neu::layer::batch_normalization bn(
			batch_size, input_dim, gamma, beta, opt, opt, queue);

		neu::gpu_vector output1(neu::layer::whole_output_size(ip), context);
		neu::gpu_vector output2(neu::layer::whole_output_size(bn), context);
		neu::layer::forward(ip, input, output1, queue);
		neu::layer::forward(bn, output1, output2, queue);

		neu::gpu_vector delta2(output2.size(), context);
		neu::range::calc_last_layer_delta(output2, teach, delta2, queue);

		neu::gpu_vector delta1(output1.size(), context);
		neu::layer::backward(bn, delta2, delta1, queue);
		neu::layer::backward_top(ip, delta1, queue);

		neu::layer::update(ip, queue);
		neu::layer::update(bn, queue);

		for(int i = 0; i < static_cast<int>(weight.size()); ++i) {
			const double numeric_grad = ip.del_weight(queue)[i];

			const auto theta = weight[i];
			const double eps = 1.0e-2f;

			const double analytic_grad = neu::calc_analytic_gradient(
				[this, i, batch_size, &weight, &gamma, &beta, &opt, &input, &teach]
				(neu::scalar theta_dash) {
					auto weight_temp = weight;
					weight_temp[i] = theta_dash;
					neu::layer::inner_product ip(
						input_dim, input_dim, batch_size, weight_temp, opt, queue);
					neu::layer::batch_normalization bn(
						batch_size, input_dim, gamma, beta, opt, opt, queue);
					neu::gpu_vector output1(neu::layer::whole_output_size(ip), context);
					neu::gpu_vector output2(neu::layer::whole_output_size(bn), context);
					neu::layer::forward(ip, input, output1, queue);
					neu::layer::forward(bn, output1, output2, queue);
					return neu::range::half_square_error_sum(output2, teach, queue)/batch_size;
				}, theta, eps);
			const auto relative_error = neu::calc_relative_error(numeric_grad, analytic_grad);
			if(relative_error > 1.0e-1f) {
				std::cout << "num: " << numeric_grad << std::endl;
				std::cout << "ana: " << analytic_grad << std::endl;
				std::cout << "relative_error: " << relative_error << std::endl;
			}
			BOOST_CHECK(numeric_grad*analytic_grad >= 0.f);
			BOOST_CHECK(relative_error < 1.0e-1f);
		}
	}
}
BOOST_AUTO_TEST_SUITE_END()
