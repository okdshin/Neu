#ifndef NEU_NODE_TRAITS_HPP
#define NEU_NODE_TRAITS_HPP
#include <boost/compute/command_queue.hpp>
#include <neu/range/gpu_buffer_range.hpp>
#include <neu/node/any_node_fwd.hpp>

namespace neu {
	namespace node {
		namespace traits {
			template<typename T>
			class add_prev {
			public:
				static decltype(auto) call(T & t, any_node* node)  {
					 t.add_prev(node);
				}
			};
		}
		template<typename T>
		decltype(auto) add_prev(T & t, any_node* node)  {
			 ::neu::node::traits::add_prev<T>::call(t, node);
		}
	}
}

namespace neu {
	namespace node {
		namespace traits {
			template<typename T>
			class add_next {
			public:
				static decltype(auto) call(T & t, any_node* node)  {
					 t.add_next(node);
				}
			};
		}
		template<typename T>
		decltype(auto) add_next(T & t, any_node* node)  {
			 ::neu::node::traits::add_next<T>::call(t, node);
		}
	}
}

namespace neu {
	namespace node {
		namespace traits {
			template<typename T>
			class forward {
			public:
				static decltype(auto) call(T & t, any_node* self, boost::compute::command_queue& queue)  {
					 t.forward(self, queue);
				}
			};
		}
		template<typename T>
		decltype(auto) forward(T & t, any_node* self, boost::compute::command_queue& queue)  {
			 ::neu::node::traits::forward<T>::call(t, self, queue);
		}
	}
}

namespace neu {
	namespace node {
		namespace traits {
			template<typename T>
			class backward {
			public:
				static decltype(auto) call(T & t, any_node* self, boost::compute::command_queue& queue)  {
					 t.backward(self, queue);
				}
			};
		}
		template<typename T>
		decltype(auto) backward(T & t, any_node* self, boost::compute::command_queue& queue)  {
			 ::neu::node::traits::backward<T>::call(t, self, queue);
		}
	}
}

namespace neu {
	namespace node {
		namespace traits {
			template<typename T>
			class update {
			public:
				static decltype(auto) call(T & t, boost::compute::command_queue& queue)  {
					 t.update(queue);
				}
			};
		}
		template<typename T>
		decltype(auto) update(T & t, boost::compute::command_queue& queue)  {
			 ::neu::node::traits::update<T>::call(t, queue);
		}
	}
}

namespace neu {
	namespace node {
		namespace traits {
			template<typename T>
			class output_for {
			public:
				static decltype(auto) call(T & t, any_node* next)  {
					return t.output_for(next);
				}
			};
		}
		template<typename T>
		decltype(auto) output_for(T & t, any_node* next)  {
			return ::neu::node::traits::output_for<T>::call(t, next);
		}
	}
}

namespace neu {
	namespace node {
		namespace traits {
			template<typename T>
			class next_delta_for {
			public:
				static decltype(auto) call(T & t, any_node* prev)  {
					return t.next_delta_for(prev);
				}
			};
		}
		template<typename T>
		decltype(auto) next_delta_for(T & t, any_node* prev)  {
			return ::neu::node::traits::next_delta_for<T>::call(t, prev);
		}
	}
}

#endif // NEU_NODE_TRAITS_HPP
