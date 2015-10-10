#ifndef NEU_INDICES_HPP
#define NEU_INDICES_HPP
//20150619
#include <neu/basic_type.hpp>
namespace neu {
	decltype(auto) calc_convolution_indices(
			int input_width, int output_width, int filter_width, int stride, int pad) {
		assert("invalid output_width or update this implementation"
			&& output_width == (input_width-filter_width+1+2*pad)/stride);

		std::vector<cpu_indices> filter_indices_list_for_input(input_width*input_width);
		std::vector<cpu_indices> output_indices_list_for_input(input_width*input_width);

		std::vector<cpu_indices> filter_indices_list_for_output(output_width*output_width);
		std::vector<cpu_indices> input_indices_list_for_output(output_width*output_width);

		std::vector<cpu_indices> input_indices_list_for_filter(filter_width*filter_width);
		std::vector<cpu_indices> output_indices_list_for_filter(filter_width*filter_width);

		for(auto _or = 0; _or < output_width; ++_or) {
			for(auto oc = 0; oc < output_width; ++oc) {
				for(auto fr = 0; fr < filter_width; ++fr) {
					for(auto fc = 0; fc < filter_width; ++fc) {
						const auto ir = _or*stride+fr-pad;
						const auto ic =  oc*stride+fc-pad;

						if(0 <= ir && ir < input_width && 0 <= ic && ic < input_width) {
							const auto input_index = ir*input_width+ic;
							const auto output_index = _or*output_width+oc;
							const auto filter_index = fr*filter_width+fc;
							filter_indices_list_for_output[output_index].push_back(filter_index);
							input_indices_list_for_output[output_index].push_back(input_index);

							filter_indices_list_for_input[input_index].push_back(filter_index);
							output_indices_list_for_input[input_index].push_back(output_index);

							input_indices_list_for_filter[filter_index].push_back(input_index);
							output_indices_list_for_filter[filter_index].push_back(output_index);
						}
					}
				}
			}
		}
		return std::make_tuple(
			input_indices_list_for_output, filter_indices_list_for_output,
			output_indices_list_for_input, filter_indices_list_for_input,
			input_indices_list_for_filter, output_indices_list_for_filter);
	}
	decltype(auto) concat_indices(std::vector<cpu_indices> const& indices_list) {
		cpu_indices concatinated;
		for(auto const& indices : indices_list) {
			concatinated.insert(concatinated.end(), indices.begin(), indices.end());
		}
		return concatinated;
	}
	decltype(auto) make_range_list(std::vector<cpu_indices> const& indices_list) {
		cpu_indices range_list;
		range_list.push_back(0);
		auto index = 0u;
		for(auto const& indices : indices_list) {
			index += indices.size();
			range_list.push_back(index);
		}
		return range_list;
	}

	struct convolution_indices {
		gpu_indices indices_range_list_for_output;
		gpu_indices input_indices_list_for_output;
		gpu_indices filter_indices_list_for_output;

		gpu_indices indices_range_list_for_input;
		gpu_indices output_indices_list_for_input;
		gpu_indices filter_indices_list_for_input;

		gpu_indices indices_range_list_for_filter;
		gpu_indices input_indices_list_for_filter;
		gpu_indices output_indices_list_for_filter;
	};

	decltype(auto) make_convolution_indices(
			int input_width, int output_width, int filter_width, int stride, int pad) {
		auto indices_tuple =
			neu::calc_convolution_indices(input_width, output_width, filter_width,
				stride, pad);

		std::cout << "here" << std::get<0>(indices_tuple).size() << std::endl;
		std::cout << "here" << std::get<1>(indices_tuple).size() << std::endl;
		std::cout << "here" << std::get<2>(indices_tuple).size() << std::endl;
		std::cout << "here" << std::get<3>(indices_tuple).size() << std::endl;
		std::cout << "here" << std::get<4>(indices_tuple).size() << std::endl;
		std::cout << "here" << std::get<5>(indices_tuple).size() << std::endl;
		auto indices_range_list_for_output = neu::to_gpu_indices(
			neu::make_range_list(std::get<0>(indices_tuple)));
		auto input_indices_list_for_output= neu::to_gpu_indices(
			neu::concat_indices(std::get<0>(indices_tuple)));
		auto filter_indices_list_for_output = neu::to_gpu_indices(
			neu::concat_indices(std::get<1>(indices_tuple)));

		auto indices_range_list_for_input = neu::to_gpu_indices(
			neu::make_range_list(std::get<2>(indices_tuple)));
		auto output_indices_list_for_input = neu::to_gpu_indices(
			neu::concat_indices(std::get<2>(indices_tuple)));
		auto filter_indices_list_for_input = neu::to_gpu_indices(
			neu::concat_indices(std::get<3>(indices_tuple)));

		auto indices_range_list_for_filter = neu::to_gpu_indices(
			neu::make_range_list(std::get<4>(indices_tuple)));
		auto input_indices_list_for_filter = neu::to_gpu_indices(
			neu::concat_indices(std::get<4>(indices_tuple)));
		auto output_indices_list_for_filter = neu::to_gpu_indices(
			neu::concat_indices(std::get<5>(indices_tuple)));

		return convolution_indices{
			std::move(indices_range_list_for_output),
			std::move(input_indices_list_for_output),
			std::move(filter_indices_list_for_output),

			std::move(indices_range_list_for_input),
			std::move(output_indices_list_for_input),
			std::move(filter_indices_list_for_input),

			std::move(indices_range_list_for_filter),
			std::move(input_indices_list_for_filter),
			std::move(output_indices_list_for_filter)
		};
	}

}// namespace neu

#endif //NEU_INDICES_HPP
