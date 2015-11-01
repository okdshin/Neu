#ifndef NEU_RANGE_TRAITS_HPP
#define NEU_RANGE_TRAITS_HPP
//20151030
#include <iterator>
#include <type_traits>

/*
namespace neu_range_traits {
	template<typename Range>
	class iterator {
	public:
		using type = typename Range::iterator;
	};
}
namespace neu {
	template<typename Range>
	using iterator_t = typename neu_range_traits::iterator<Range>::type;
}

namespace neu_range_traits {
	template<typename Range>
	class value {
	public:
		using type = typename std::iterator_traits<neu::iterator_t<Range>>::value_type;
	};
}
namespace neu {
	template<typename Range>
	using value_t = typename neu_range_traits::value<Range>::type;
}
*/

namespace neu_range_traits {
	template<typename Range>
	class begin {
	public:
		static decltype(auto) call(Range const& range) {
			return range.begin();
		}
	};
	template<typename Range>
	class end {
	public:
		static decltype(auto) call(Range const& range) {
			return range.end();
		}
	};
}
namespace neu {
	template<typename Range>
	decltype(auto) range_begin(Range const& range) {
		return neu_range_traits::begin<std::decay_t<Range>>::call(range);
	}
	template<typename Range>
	decltype(auto) range_end(Range const& range) {
		return neu_range_traits::end<std::decay_t<Range>>::call(range);
	}
}

namespace neu_range_traits {
	template<typename Range>
	class distance {
	public:
		static decltype(auto) call(Range const& range) {
			return static_cast<std::size_t>(
				std::distance(neu::range_begin(range), neu::range_end(range)));
		}
	};
}
namespace neu {
	template<typename Range>
	decltype(auto) range_distance(Range const& range) {
		return neu_range_traits::distance<std::decay_t<Range>>::call(range);
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
