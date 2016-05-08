#ifndef NEU_LAYER_READY_MADE_DEEPCNET_HPP
#define NEU_LAYER_READY_MADE_DEEPCNET_HPP
//20160129
#include <neu/layer/any_layer.hpp>
#include <neu/layer/any_layer_vector.hpp>
#include <neu/layer/convolution.hpp>
#include <neu/layer/convolution_optimized.hpp>
#include <neu/layer/convolution_optimized_wr.hpp>
#include <neu/layer/inner_product_wr.hpp>
#include <neu/layer/max_pooling.hpp>
#include <neu/layer/activation/leaky_rectifier.hpp>
#include <neu/layer/bias.hpp>
#include <neu/layer/shared_dropout.hpp>
#include <neu/layer/batch_normalization.hpp>
#include <neu/layer/activation/softmax_loss.hpp>
#include <neu/layer/dropout_cpu.hpp>
namespace neu {
	namespace layer {
		namespace ready_made {
			template<typename Rng, typename OptGen>
			decltype(auto) make_deepcnet(
					std::vector<neu::layer::any_layer>& nn,
					int batch_size, int input_width, int label_num,
					int l, int k,
					scalar dropout_base_probability,
					Rng&& g, OptGen const& optgen,
					boost::compute::command_queue& queue) {
				scalar c = 1.f;
				std::cout << "dro:" << dropout_base_probability << std::endl;
				for(int li = 0; li < l+1; ++li) {
					const auto glp = [&nn, li, input_width, k]() {
						if(li == 0) {
							return geometric_layer_property{
								input_width, 3, 3,
								(li+1)*k, 1, input_width};
						}
						else {
							return geometric_layer_property{
								output_width(nn), 2, output_channel_num(nn),
								(li+1)*k, 1, 0};
						}
					}();

					// dropout_cpu
					nn.push_back(neu::layer::dropout_cpu(
						neu::layer::input_dim(glp),
						batch_size,
						1.f-dropout_base_probability*li,
						queue));
					// dropout
					/*
					nn.push_back(neu::layer::dropout(
						neu::layer::input_dim(glp),
						batch_size,
						1.f-dropout_base_probability*li,
						queue));
					*/
					// shared_dropout
					/*
					nn.push_back(neu::layer::shared_dropout(
						batch_size,
						neu::layer::input_dim(glp),
						glp.input_width*glp.input_width,
						1.f-dropout_base_probability*li,
						queue));
					std::cout << li << ": dropout: " << (1.f-dropout_base_probability*li) << std::endl;
					*/

					// conv
					nn.push_back(make_convolution_optimized_wr_xavier(
						glp, batch_size, c, g,
						optgen(neu::layer::filters_size(glp)), queue));

					auto output_width = neu::layer::output_width(nn.back());
					auto output_channel_num = neu::layer::output_channel_num(nn.back());

					// bias //TODO shared
					/*
					nn.push_back(neu::layer::make_bias(
						output_dim(nn), batch_size, [](){ return 0; },
						optgen(output_dim(nn)), queue));
					*/
					/*
					// bn
					nn.push_back(neu::layer::make_batch_normalization(
						batch_size, neu::layer::output_dim(nn),
						optgen(output_dim(nn)),
						optgen(output_dim(nn)),
						queue));
					*/

					if(li != l) {
						// leaky relu
						nn.push_back(neu::layer::make_leaky_rectifier(
							neu::layer::output_dim(nn), batch_size, 0.33f));
					}

					if(li != l) {
						// max pooling
						neu::layer::geometric_layer_property glp{
							output_width,
							2,
							output_channel_num,
							output_channel_num,
							2, 0
						};
						nn.push_back(neu::layer::max_pooling(
							glp, batch_size, queue.get_context()));
					}
				}

				// inner_product
				nn.push_back(neu::layer::make_inner_product_wr_xavier(
					neu::layer::output_dim(nn), label_num, batch_size, c, g,
					optgen(neu::layer::output_dim(nn)*label_num),
					queue));

				// bias //TODO shared
				/*
				nn.push_back(neu::layer::make_bias(
					output_dim(nn), batch_size, [](){ return 0; },
					optgen(output_dim(nn)), queue));
				*/
				/*
				// bn
				nn.push_back(neu::layer::make_batch_normalization(
					batch_size, neu::layer::output_dim(nn),
					optgen(output_dim(nn)),
					optgen(output_dim(nn)),
					queue));
				*/

				// softmax_loss
				nn.push_back(neu::layer::make_softmax_loss(
					neu::layer::output_dim(nn), batch_size, queue.get_context()));

				return nn;
			}
		}
	}
}// namespace neu

#endif //NEU_LAYER_READY_MADE_DEEPCNET_HPP
