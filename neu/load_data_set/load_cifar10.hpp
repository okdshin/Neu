#ifndef NEU_DATA_SET_LOADER_LOAD_CIFAR10_HPP
#define NEU_DATA_SET_LOADER_LOAD_CIFAR10_HPP
//20151008
#include <vector>
#include <cassert>
#include <algorithm>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	decltype(auto) load_cifar10_train_data(boost::filesystem::path const& dir) {
		constexpr auto label_num = 10u;
		constexpr auto input_dim = 1024u*3u;
		constexpr auto data_num_per_file = 10000u;
		std::vector<std::vector<cpu_vector>> data(label_num);
		for(auto i = 1u; i <= 5u; ++i) {
			boost::filesystem::path filename("data_batch_"+std::to_string(i)+".bin");
			std::cout << filename << std::endl;
			std::cout << dir/filename << std::endl;
			boost::filesystem::ifstream ifs(dir/filename, std::ios::binary);
			for(auto i = 0u; i < data_num_per_file; ++i) {
				assert(ifs);
				char label_id;
				ifs.read(&label_id, 1);
				std::vector<char> input_byte(input_dim);
				ifs.read(input_byte.data(), input_dim);
				std::vector<scalar> input(input_dim);
				std::transform(input_byte.begin(), input_byte.end(), input.begin(),
					[](auto e){
						return static_cast<scalar>(static_cast<unsigned char>(e)); });
				data[label_id].push_back(std::move(input));
			}
		}
		return data;
	}
	decltype(auto) load_cifar10_test_data(boost::filesystem::path const& dir) {
		constexpr auto label_num = 10u;
		constexpr auto input_dim = 1024u*3u;
		constexpr auto data_num  = 10000u;
		std::vector<std::vector<cpu_vector>> data(label_num);
		boost::filesystem::path filename("test_batch.bin");
		std::cout << filename << std::endl;
		std::cout << dir/filename << std::endl;
		boost::filesystem::ifstream ifs(dir/filename, std::ios::binary);
		for(auto i = 0u; i < data_num; ++i) {
			assert(ifs);
			char label_id;
			ifs.read(&label_id, 1);
			std::vector<char> input_byte(input_dim);
			ifs.read(input_byte.data(), input_dim);
			std::vector<scalar> input(input_dim);
			std::transform(input_byte.begin(), input_byte.end(), input.begin(),
				[](auto e){
					return static_cast<scalar>(static_cast<unsigned char>(e)); });
			data[label_id].push_back(std::move(input));
		}
		return data;
	}
}// namespace neu

#endif //NEU_DATA_SET_LOADER_LOAD_CIFAR10_HPP
