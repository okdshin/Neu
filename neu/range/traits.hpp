#ifndef NEU_RANGE_TRAITS_HPP
#define NEU_RANGE_TRAITS_HPP
//20151030
#include <iterator>
#include <type_traits>
#include <neu/basic_type.hpp>

namespace neu {
	namespace range {
		namespace traits {
			template<typename Range>
			class begin {
			public:
				static decltype(auto) call(Range const& range) {
					return range.begin();
				}
				static decltype(auto) call(Range& range) {
					return range.begin();
				}
			};
			template<typename Range>
			class end {
			public:
				static decltype(auto) call(Range const& range) {
					return range.end();
				}
				static decltype(auto) call(Range& range) {
					return range.end();
				}
			};
		}
		template<typename Range>
		decltype(auto) begin(Range& range) {
			return ::neu::range::traits::begin<std::decay_t<Range>>::call(range);
		}
		template<typename Range>
		decltype(auto) begin(Range const& range) {
			return ::neu::range::traits::begin<std::decay_t<Range>>::call(range);
		}

		template<typename Range>
		decltype(auto) end(Range& range) {
			return ::neu::range::traits::end<std::decay_t<Range>>::call(range);
		}
		template<typename Range>
		decltype(auto) end(Range const& range) {
			return ::neu::range::traits::end<std::decay_t<Range>>::call(range);
		}

		namespace traits {
			template<typename Range>
			class distance {
			public:
				static decltype(auto) call(Range const& range) {
					return static_cast<int>(
						std::distance(::neu::range::begin(range), ::neu::range::end(range)));
				}
			};
		}
		template<typename Range>
		decltype(auto) distance(Range const& range) {
			return ::neu::range::traits::distance<std::decay_t<Range>>::call(range);
		}

		namespace traits {
			template<typename Range>
			class get_buffer {
			public:
				static decltype(auto) call(Range const& range) {
					return range.get_buffer();
				}
			};
		}
		template<typename Range>
		decltype(auto) get_buffer(Range const& range) {
			return ::neu::range::traits::get_buffer<std::decay_t<Range>>::call(range);
		}

		namespace traits {
			template<typename Range>
			class get_begin_index {
			public:
				static decltype(auto) call(Range const& range) {
					return range.get_begin_index();
				}
			};
			template<typename T>
			class get_begin_index<boost::compute::vector<T>> {
			public:
				static decltype(auto) call(boost::compute::vector<T> const& range) {
					return range.begin().get_index();
				}
			};
		}
		template<typename Range>
		decltype(auto) get_begin_index(Range const& range) {
			return ::neu::range::traits::get_begin_index<std::decay_t<Range>>::call(range);
		}
	}
}

#endif //NEU_RANGE_TRAITS_HPP
