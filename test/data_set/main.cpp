#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <neu/max_pooling_layer.hpp>
#include <neu/image.hpp>
#include <neu/kernel.hpp>
#include <neu/data_set.hpp>
#include <neu/vector_io.hpp>

int main() {
	std::cout << "hello world" << std::endl;
	std::mt19937 rand(0);
	std::vector<std::vector<neu::cpu_vector>> data = {
		{
			{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}
		},
		{
			{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}
		},
		{
			{2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}
		}
	};
	auto ds = neu::make_data_set(3, 4, 2, data, rand);
	{
		neu::make_next_batch(ds);
		auto batch = ds.get_batch();
		neu::print(batch.train_data);
		neu::print(batch.teach_data);
	}
	{
		auto f = ds.async_make_next_batch();
		std::cout << "hello" << std::endl;
		f.wait();
		auto batch = ds.get_batch();
		neu::print(batch.train_data);
		neu::print(batch.teach_data);
	}
}
