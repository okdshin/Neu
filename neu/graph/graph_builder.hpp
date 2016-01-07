#ifndef NEU_GRAPH_GRAPH_BUILDER_HPP
#define NEU_GRAPH_GRAPH_BUILDER_HPP
//20151230
#include <string>
#include <map>
#include <neu/graph/node.hpp>
namespace neu {
	namespace graph {
		class graph_builder {
		public:
			template<typename Node>
			decltype(auto) add_node(std::string const& id, Node const& node) {
				if(node_map_.find(id) != node_map_.end()) {
					throw "same id node already exist";
				}
				node_map_[id] = std::make_unique<Node>(node);
			}

			decltype(auto) connect(std::string const& from, std::string const& to) {
				graph::connect(*node_map_[from], *node_map_[to]);
			}

			/*
			decltype(auto) build() const {
				return layer::deep_layer(node_map_);
			}
			*/

			decltype(auto) operator[](std::string const& id) {	
				return *(node_map_[id]);
			}

		private:
			std::map<std::string, std::unique_ptr<node>> node_map_;
		};

		decltype(auto) connect_flow(graph_builder& builder,
				std::vector<std::string> const& id_list) {
			for(auto i = 1; i < static_cast<int>(id_list.size()); ++i) {
				builder.connect(id_list[i-1], id_list[i]);
			}
		}


	}
}// namespace neu

#endif //NEU_GRAPH_GRAPH_BUILDER_HPP
