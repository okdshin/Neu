#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <neu/image.hpp>

int main(int argc, char** argv) {
	namespace bpo = boost::program_options;

	std::cout << "filter_extractor" << std::endl;
	bpo::options_description option("option");
	std::string input_path, output_dir_path;
	option.add_options()
		("help,h", "show help")
		("input,i", bpo::value<std::string>(&input_path)->default_value(
			"../../../data/learned_mnist_conv.yaml"), "learned nn yaml file path")
		("output_dir,odir", bpo::value<std::string>(&output_dir_path)->default_value(
			"./result/"), "output filter image dir path")
	;
	bpo::variables_map argmap;
	bpo::store(bpo::parse_command_line(argc, argv, option), argmap);
	bpo::notify(argmap);
	if (argmap.count("help")) {
		std::cout << option << "\n";
		return 1;
	}
	if(output_dir_path.back() != '/') {
		std::cout << "last charactor of output dir path should be /" << std::endl;
		return 1;
	}

	std::cout << input_path << std::endl;
	auto nn_data = YAML::LoadFile(input_path.c_str());
	auto layers_data = nn_data["layers"];
	auto i = 0;
	for(auto layer_data : layers_data) {
		if(layer_data["layer_type"].as<std::string>() ==  "convolution") {
			neu::save_image_vector_as_images(
				layer_data["filters"].as<std::vector<float>>(),
				layer_data["filter_width"].as<std::size_t>(),
				layer_data["input_channel_num"].as<std::size_t>(),
				layer_data["output_channel_num"].as<std::size_t>(),
				"filter"+std::to_string(i)+"_.bmp", 255.f);
		}
		++i;
	}
}
