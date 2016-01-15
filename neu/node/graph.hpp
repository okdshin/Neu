#ifndef NEU_NODE_GRAPH_HPP
#define NEU_NODE_GRAPH_HPP
//20151230
#include <string>
#include <map>
#include <neu/node/any_node.hpp>
namespace neu {
	namespace node {
		decltype(auto) connect(any_node& from, any_node& to) {
			from.add_next(&to);
			to.add_prev(&from);
		}
		class graph {
		public:
			decltype(auto) add_node(std::string const& id, any_node const& node) {
				if(node_map_.find(id) != node_map_.end()) {
					throw "same id node already exist";
				}
				node_map_[id] = node;
			}

			decltype(auto) connect(std::string const& from, std::string const& to) {
				node::connect(node_map_[from], node_map_[to]);
			}

			decltype(auto) operator[](std::string const& id) {	
				return node_map_[id];
			}

		private:
			std::map<std::string, any_node> node_map_;
		};

		decltype(auto) connect_flow(graph& g,
				std::vector<std::string> const& id_list) {
			for(auto i = 1; i < static_cast<int>(id_list.size()); ++i) {
				g.connect(id_list[i-1], id_list[i]);
			}
		}
	}
}// namespace neu

#endif //NEU_NODE_GRAPH_HPP
