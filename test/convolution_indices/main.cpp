#include <iostream>
#include <neu/convolution_layer.hpp>
#include <neu/vector_io.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	auto stride = 1;
	auto input_width = 5;
	auto filter_width = 3;
	auto indices_tuple = neu::calc_indices(input_width, filter_width, stride);

	for(const auto& e : std::get<1>(indices_tuple)) {
		neu::print(e);
	}
	std::cout << "\n";
	for(const auto& e : std::get<2>(indices_tuple)) {
		neu::print(e);
	}

	auto input_indices = std::get<1>(indices_tuple);
	auto concatinated_input_indices = neu::concat_indices(input_indices);
	neu::print(concatinated_input_indices);
	auto input_indices_range_list = neu::make_range_list(input_indices);
	std::cout << "range: "; neu::print(input_indices_range_list);
}
