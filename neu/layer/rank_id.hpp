#ifndef NEU_LAYER_RANK_ID_HPP
#define NEU_LAYER_RANK_ID_HPP
//20160115

namespace neu {
	namespace layer {
		enum class rank_id : int {
			dim = 0,
			width = 0,
			height = 1,
			channel_num = 2
		};
	}
}// namespace neu

#endif //NEU_LAYER_RANK_ID_HPP
