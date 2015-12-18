#ifndef NEU_OPTIMIZER_TRAITS_HPP
#define NEU_OPTIMIZER_TRAITS_HPP
//20151212
#include <type_traits>
#include <yaml-cpp/yaml.h>

namespace neu {
	namespace optimizer {
		//
		// apply
		//
		namespace traits {
			template<typename Optimizer>
			class apply {
			public:
				static decltype(auto) call(Optimizer& o,
						gpu_vector& weight, gpu_vector const& del_weight,
						boost::compute::command_queue& queue) {
					o.apply(weight, del_weight, queue);
				}
			};
		}
		template<typename Optimizer>
		decltype(auto) apply(Optimizer& o,
				gpu_vector& weight, gpu_vector const& del_weight,
				boost::compute::command_queue& queue) {
			::neu::optimizer::traits::apply<std::decay_t<Optimizer>>::call(o,
				weight, del_weight, queue);
		}

		//
		// serialize
		//
		namespace traits {
			template<typename Optimizer>
			class serialize {
			public:
				static decltype(auto) call(Optimizer const& o,
						YAML::Emitter& emitter,
						boost::compute::command_queue& queue) {
					o.serialize(emitter, queue);
				}
			};
		}
		template<typename Optimizer>
		decltype(auto) serialize(Optimizer const& o,
				YAML::Emitter& emitter,
				boost::compute::command_queue& queue) {
			::neu::optimizer::traits::serialize<std::decay_t<Optimizer>>::call(o,
				emitter, queue);
		}
	}
}// namespace neu

#endif //NEU_OPTIMIZER_TRAITS_HPP
