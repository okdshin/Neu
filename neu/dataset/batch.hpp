#ifndef NEU_DATASET_BATCH_HPP
#define NEU_DATASET_BATCH_HPP
//20151230

namespace neu {
	namespace dataset {
		struct batch {
			gpu_vector train_data;
			gpu_vector teach_data;
		};
	}
}// namespace neu

#endif //NEU_DATASET_BATCH_HPP
