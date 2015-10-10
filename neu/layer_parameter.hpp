#ifndef NEU_LAYER_PARAMETER_HPP
#define NEU_LAYER_PARAMETER_HPP
//20150923
#include <exception>
#define NEU_PP_PARAMETER(name) \
	private: \
		std::size_t name##_; \
		bool is_##name##_set_ = false; \
	public: \
		decltype(auto) name(std::size_t name) { \
			name##_ = name; \
			is_##name##_set_ = true; \
			return *this; \
		} \
		decltype(auto) name() const { \
			if(!is_##name##_set_) { throw std::runtime_error("param is not set"); } \
			return name##_; \
		} \
		decltype(auto) name##_opt() const { return name##_; } \
		decltype(auto) is_##name##_set() const { return is_##name##_set_; }
#endif //NEU_LAYER_PARAMETER_HPP
