#ifndef NEU_LAYER_DESERIALIZE_HPP
#define NEU_LAYER_DESERIALIZE_HPP
//20151211
#include <exception>
#include <yaml-cpp/yaml.h>
#include <neu/layer/inner_product.hpp>
#include <neu/layer/bias.hpp>
#include <neu/layer/convolution.hpp>
#include <neu/layer/max_pooling.hpp>
#include <neu/layer/activation/sigmoid.hpp>
#include <neu/layer/activation/rectifier.hpp>
#include <neu/layer/activation/leaky_rectifier.hpp>
#include <neu/layer/activation/sigmoid_loss.hpp>
#include <neu/layer/activation/softmax_loss.hpp>
#include <neu/layer/dropout.hpp>
#include <neu/layer/shared_dropout.hpp>
#include <neu/layer/any_layer_vector.hpp>
namespace neu {
	namespace layer {
		class deserialize_error : public std::exception {
		public:
			explicit deserialize_error(std::string const& layer_id)
				: layer_id_(layer_id) {}

			virtual const char* what() const noexcept {
				return ("Unknown layer \""+ layer_id_
					+ "\" is tried to be deserialized. Check layer data and deserialize function.").c_str();
			}
		private:
			std::string layer_id_;
		};

		any_layer deserialize(YAML::Node const& node, 
				boost::compute::command_queue& queue) {
			auto lt = node["layer_type"].as<std::string>();
			if(lt == "inner_product") {
				return static_cast<any_layer>(deserialize_inner_product(node, queue));
			} else
			if(lt == "bias") {
				return static_cast<any_layer>(deserialize_bias(node, queue));
			} else
			if(lt == "convolution") {
				return static_cast<any_layer>(deserialize_convolution(node, queue));
			} else
			if(lt == "max_pooling") {
				return static_cast<any_layer>(deserialize_max_pooling(node, queue.get_context()));
			} else
			if(lt == "sigmoid") {
				return static_cast<any_layer>(deserialize_sigmoid(node, queue));
			} else
			if(lt == "rectifier") {
				return static_cast<any_layer>(deserialize_rectifier(node, queue));
			} else
			if(lt == "leaky_rectifier") {
				return static_cast<any_layer>(deserialize_leaky_rectifier(node, queue));
			} else
			if(lt == "sigmoid_loss") {
				return static_cast<any_layer>(deserialize_sigmoid_loss(node, queue));
			} else
			if(lt == "softmax_loss") {
				return static_cast<any_layer>(deserialize_softmax_loss(node, queue));
			} else
			if(lt == "dropout") {
				return static_cast<any_layer>(deserialize_dropout(node, queue));
			} else
			if(lt == "shared_dropout") {
				return static_cast<any_layer>(deserialize_shared_dropout(node, queue));
			} else
			if(lt == "any_layer_vector") {
				return static_cast<any_layer>(deserialize_any_layer_vector(node, queue));
			}
			else {
				throw deserialize_error(lt);
			}
		}
	}
}// namespace neu

#endif //NEU_LAYER_DESERIALIZE_HPP
