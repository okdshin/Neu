#ifndef NEU_DATA_SET_LOADER_LOAD_MNIST_HPP
#define NEU_DATA_SET_LOADER_LOAD_MNIST_HPP
//20151008
#include <vector>
#include <cassert>
#include <algorithm>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <neu/basic_type.hpp>
namespace neu {
	namespace dataset {
		decltype(auto) load_mnist(boost::filesystem::path const& dir) {
			constexpr auto label_num = 10u;
			constexpr auto input_dim = 28u*28u;
			constexpr auto data_num_per_file = 60000u;
			constexpr auto label_header_size = 8u;
			constexpr auto image_header_size = 16u;

			boost::filesystem::path label_filename("train-labels-idx1-ubyte");
			std::cout << "label: " << dir/label_filename << std::endl;
			boost::filesystem::ifstream label_ifs(dir/label_filename, std::ios::binary);
			label_ifs.ignore(label_header_size);

			boost::filesystem::path image_filename("train-images-idx3-ubyte");
			std::cout << "image: " << dir/image_filename << std::endl;
			boost::filesystem::ifstream image_ifs(dir/image_filename, std::ios::binary);
			image_ifs.ignore(image_header_size);

			std::vector<std::vector<cpu_vector>> data(label_num);
			for(auto i = 0u; i < data_num_per_file; ++i) {
				assert(label_ifs && image_ifs);
				char label_id;
				label_ifs.read(&label_id, 1);
				std::vector<char> input_byte(input_dim);
				image_ifs.read(input_byte.data(), input_dim);
				std::vector<scalar> input(input_dim);
				std::transform(input_byte.begin(), input_byte.end(), input.begin(),
					[](auto e){
						return static_cast<scalar>(static_cast<unsigned char>(e)); });
				data[label_id].push_back(std::move(input));
			}
			return data;
		}
	}
}// namespace neu

#endif //NEU_DATA_SET_LOADER_LOAD_MNIST_HPP
