#include <iostream>
#include <neu/vector_io.hpp>
#include <boost/compute/algorithm/accumulate.hpp>
#include <neu/activation_func/rectifier.hpp>
#include <neu/activation_func/sigmoid.hpp>
#include <neu/activation_func/softmax.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	auto softmax = neu::softmax(3, 3);
	auto x = neu::to_gpu_vector(neu::cpu_vector{-1.68e+13, -1.56e+13, -5.705e+12, 3., 4., 5., 88., 0., 0.});
	auto y = softmax(x);
	neu::print(x);
	neu::print(y);
	auto sum = boost::compute::accumulate(y.begin(), y.end(), 0.f, boost::compute::plus<neu::scalar>());
	std::cout << "y_sum: " << sum << std::endl;
}
