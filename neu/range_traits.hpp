#ifndef NEU_RANGE_TRAITS_HPP
#define NEU_RANGE_TRAITS_HPP
//20151030
#include <iterator>
#include <type_traits>
namespace neu_range_traits {
	template<typename Range>
	class size {
	public:
		static decltype(auto) call(Range const& range) {
			return range.size();
		}
	};
}

namespace neu {
	template<typename Range>
	decltype(auto) range_size(Range const& range) {
		return neu_range_traits::size<std::decay_t<Range>>::call(range);
	}
}

namespace neu_range_traits {
	template<typename Range>
	class get_buffer {
	public:
		static decltype(auto) call(Range const& range) {
			return range.get_buffer();
		}
	};
}

namespace neu {
	template<typename Range>
	decltype(auto) range_get_buffer(Range const& range) {
		return neu_range_traits::get_buffer<std::decay_t<Range>>::call(range);
	}
}

namespace neu_range_traits {
	template<typename Range>
	class get_begin_index {
	public:
		static decltype(auto) call(Range const& range) {
			return range.begin().get_index();
		}
	};
	template<typename Range>
	class get_end_index {
	public:
		static decltype(auto) call(Range const& range) {
			return range.end().get_index();
		}
	};
}

namespace neu {
	template<typename Range>
	decltype(auto) range_get_begin_index(Range const& range) {
		return neu_range_traits::get_begin_index<std::decay_t<Range>>::call(range);
	}
	template<typename Range>
	decltype(auto) range_get_end_index(Range const& range) {
		return neu_range_traits::get_end_index<std::decay_t<Range>>::call(range);
	}
}
#endif //NEU_RANGE_TRAITS_HPP
