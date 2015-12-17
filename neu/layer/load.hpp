#ifndef NEU_LAYER_LOAD_HPP
#define NEU_LAYER_LOAD_HPP
//20151211
#include <exception>
#include <yaml-cpp/yaml.h>
#include <neu/layer/inner_product.hpp>
#include <neu/layer/bias.hpp>
#include <neu/layer/activation/sigmoid.hpp>
#include <neu/layer/activation/rectifier.hpp>
#include <neu/layer/activation/sigmoid_loss.hpp>
#include <neu/layer/any_layer_vector.hpp>
namespace neu {
	namespace layer {
		class load_error : public std::exception {
		public:
			explicit load_error(std::string const& layer_id) : layer_id_(layer_id) {}

			virtual const char* what() const noexcept {
				return ("Unknown layer \""+ layer_id_
					+ "\" is loaded. Check layer data and load function.").c_str();
			}
		private:
			std::string layer_id_;
		};
		any_layer load(YAML::Node const& node, 
				boost::compute::command_queue& queue) {
			auto lt = node["layer_type"].as<std::string>();
			if(lt == "inner_product") {
				return static_cast<any_layer>(load_inner_product(node, queue));
			} else
			if(lt == "bias") {
				return static_cast<any_layer>(load_bias(node, queue));
			} else
			if(lt == "sigmoid") {
				return static_cast<any_layer>(load_sigmoid(node, queue));
			} else
			if(lt == "rectifier") {
				return static_cast<any_layer>(load_rectifier(node, queue));
			} else
			if(lt == "leaky_rectifier") {
				return static_cast<any_layer>(load_leaky_rectifier(node, queue));
			} else
			if(lt == "sigmoid_loss") {
				return static_cast<any_layer>(load_sigmoid_loss(node, queue));
			} else
			if(lt == "any_layer_vector") {
				return static_cast<any_layer>(load_any_layer_vector(node, queue));
			}
			else {
				throw load_error(lt);
			}
		}
	}
}// namespace neu

#endif //NEU_LAYER_LOAD_HPP
