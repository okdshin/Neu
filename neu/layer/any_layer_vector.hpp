#ifndef NEU_ANY_LAYER_VECTOR_HPP
#define NEU_ANY_LAYER_VECTOR_HPP
//20150914
#include <vector>
#include <neu/basic_type.hpp>
#include <neu/range/algorithm.hpp>
#include <neu/range/gpu_buffer_range.hpp>
#include <neu/layer/traits.hpp>
#include <neu/layer/any_layer.hpp>
#include <neu/layer/deserialize_fwd.hpp>
namespace neu {
	namespace layer {
		namespace traits {
			template<>
			class input_rank<std::vector<neu::layer::any_layer>> {
			public:
				static decltype(auto) call(
						std::vector<neu::layer::any_layer> const& alv) {
					return ::neu::layer::input_rank(alv.front());
				}
			};
			template<>
			class output_rank<std::vector<neu::layer::any_layer>> {
			public:
				static decltype(auto) call(
						std::vector<neu::layer::any_layer> const& alv) {
					return ::neu::layer::output_rank(alv.back());
				}
			};
			template<>
			class input_size<std::vector<neu::layer::any_layer>> {
			public:
				static decltype(auto) call(
						std::vector<neu::layer::any_layer> const& alv, rank_id ri) {
					return ::neu::layer::input_size(alv.front(), ri);
				}
			};
			template<>
			class output_size<std::vector<neu::layer::any_layer>> {
			public:
				static decltype(auto) call(
						std::vector<neu::layer::any_layer> const& alv, rank_id ri) {
					return ::neu::layer::output_size(alv.back(), ri);
				}
			};
			template<>
			class batch_size<std::vector<neu::layer::any_layer>> {
			public:
				static decltype(auto) call(
						std::vector<neu::layer::any_layer> const& alv) {
					return ::neu::layer::batch_size(alv.front());
				}
			};
			template<>
			class test_forward<std::vector<neu::layer::any_layer>> {
			public:
				template<typename InputRange, typename OutputRange>
				static decltype(auto) call(
						std::vector<neu::layer::any_layer>& layers,
						std::size_t batch_size,
						InputRange const& initial_input, OutputRange& result_output,
						boost::compute::command_queue& queue) {
					gpu_vector input(initial_input.begin(), initial_input.end(), queue);
					gpu_vector output(queue.get_context());
					for(auto& l : layers) {
						output.resize(::neu::layer::whole_output_size(l), queue);
						l.test_forward(batch_size,
							range::to_range(input), range::to_range(output), queue);
						input.swap(output);
					}
					range::copy(input, result_output, queue);
				}
			};
			template<>
			class forward<std::vector<neu::layer::any_layer>> {
			public:
				template<typename InputRange, typename OutputRange>
				static decltype(auto) call(std::vector<neu::layer::any_layer>& layers,
						InputRange const& initial_input, OutputRange& result_output,
						boost::compute::command_queue& queue) {
					gpu_vector input(initial_input.begin(), initial_input.end(), queue);
					gpu_vector output(queue.get_context());
					for(auto& l : layers) {
						output.resize(::neu::layer::whole_output_size(l), queue);
						l.forward(range::to_range(input), range::to_range(output), queue);
						input.swap(output);
					}
					range::copy(input, result_output, queue);
				}
			};
			template<>
			class backward_top<std::vector<neu::layer::any_layer>> {
			public:
				template<typename InputRange>
				static decltype(auto) call(std::vector<neu::layer::any_layer>& layers,
						InputRange const& initial_delta,
						boost::compute::command_queue& queue) {
					gpu_vector delta(initial_delta.begin(), initial_delta.end(), queue);
					gpu_vector prev_delta(queue.get_context());

					// call backward except the top layer
					for(int i = layers.size()-1; i >= 1; --i) {
						auto& l = layers.at(i);
						prev_delta.resize(::neu::layer::whole_input_size(l), queue);
						l.backward(
							range::to_range(delta), range::to_range(prev_delta),
							queue);
						delta.swap(prev_delta);
					}

					// call backward_top on the top layer
					layers.front().backward_top(range::to_range(delta), queue);
				}
			};
			template<>
			class backward<std::vector<neu::layer::any_layer>> {
			public:
				template<typename InputRange, typename OutputRange>
				static decltype(auto) call(std::vector<neu::layer::any_layer>& layers,
						InputRange const& initial_delta,
						OutputRange& result_prev_delta,
						boost::compute::command_queue& queue) {
					gpu_vector delta(initial_delta.begin(), initial_delta.end(), queue);
					gpu_vector prev_delta(queue.get_context());
					for(int i = layers.size()-1; i >= 0; --i) {
						auto& l = layers.at(i);
						prev_delta.resize(::neu::layer::whole_input_size(l), queue);
						l.backward(
							range::to_range(delta), range::to_range(prev_delta),
							queue);
						delta.swap(prev_delta);
					}
					range::copy(delta, result_prev_delta, queue);
				}
			};
			template<>
			class update<std::vector<neu::layer::any_layer>> {
			public:
				static decltype(auto) call(std::vector<neu::layer::any_layer>& layers,
						boost::compute::command_queue& queue) {
					for(auto& l : layers) {
						l.update(queue);
					}
				}
			};
			template<>
			class serialize<std::vector<neu::layer::any_layer>> {
			public:
				static decltype(auto) call(
						std::vector<neu::layer::any_layer> const& layers,
						YAML::Emitter& emitter, boost::compute::command_queue& queue) {
					emitter << YAML::BeginMap
						<< YAML::Key << "layer_type"
							<< YAML::Value << "any_layer_vector"
						<< YAML::Key << "input_dim"
							<< YAML::Value << ::neu::layer::input_dim(layers)
						<< YAML::Key << "output_dim"
							<< YAML::Value << ::neu::layer::output_dim(layers)
						<< YAML::Key << "batch_size"
							<< YAML::Value << ::neu::layer::batch_size(layers)
						<< YAML::Key << "layers" << YAML::Value
							<< YAML::BeginSeq;
					for(auto const& l : layers) {
						::neu::layer::serialize(l, emitter, queue);
					}
					emitter << YAML::EndSeq << YAML::EndMap;
				}
			};
		}
		decltype(auto) deserialize_any_layer_vector(YAML::Node const& node,
				boost::compute::command_queue& queue) {
			NEU_ASSERT(node["layer_type"].as<std::string>() == "any_layer_vector");
			auto layers_node = node["layers"];
			std::vector<neu::layer::any_layer> layers;
			for(auto const& ln : layers_node) {
				layers.push_back(::neu::layer::deserialize(ln, queue));
			}
			return layers;
		}
	}
}// namespace neu

#endif //NEU_ANY_LAYER_VECTOR_HPP
