#ifndef NEU_LAYER_READY_MADE_DEEPCNET_HPP
#define NEU_LAYER_READY_MADE_DEEPCNET_HPP
//20160129
#include <neu/layer/any_layer.hpp>
#include <neu/layer/any_layer_vector.hpp>
#include <neu/layer/convolution.hpp>
#include <neu/layer/max_pooling.hpp>
#include <neu/layer/activation/leaky_rectifier.hpp>
#include <neu/layer/bias.hpp>
namespace neu {
	namespace layer {
		namespace ready_made {
			template<typename Rng, typename OptGen>
			decltype(auto) make_deepcnet(
					std::vector<neu::layer::any_layer>& nn,
					int batch_size, int input_width, int l, int k, Rng&& g,
					OptGen const& optgen,
					boost::compute::command_queue& queue) {
				for(int li = 0; li < l+1; ++li) {
					// conv
					const auto glp = [&nn, li, input_width, k]() {
						if(li == 0) {
							return geometric_layer_property{
								input_width, 3, 3,
								(li+1)*k, 1, 1};
						}
						else {
							return geometric_layer_property{
								output_width(nn), 2, output_channel_num(nn),
								(li+1)*k, 1, 1};
						}
					}();
					nn.push_back(make_convolution_xavier(
						glp, batch_size, g, optgen(neu::layer::filters_size(glp)), queue));
					auto output_width = neu::layer::output_width(nn.back());
					auto output_channel_num = neu::layer::output_channel_num(nn.back());

					// bias
					nn.push_back(neu::layer::make_bias(
						output_dim(nn), batch_size, [](){ return 0; },
						optgen(output_dim(nn)), queue));

					// leaky relu
					nn.push_back(neu::layer::make_leaky_rectifier(
						neu::layer::output_dim(nn), batch_size, 0.33f));

					if(li != l) {
						// max pooling
						neu::layer::geometric_layer_property glp{
							output_width,
							2,
							output_channel_num,
							output_channel_num,
							2, 1
						};
						nn.push_back(neu::layer::max_pooling(glp, batch_size, queue.get_context()));
					}
				}
				return nn;
			}
		}
	}
}// namespace neu

#endif //NEU_LAYER_READY_MADE_DEEPCNET_HPP
