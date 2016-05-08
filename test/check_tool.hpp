#ifndef NEU_TEST_CHECK_TOOL_HPP
#define NEU_TEST_CHECK_TOOL_HPP
#include <boost/test/unit_test.hpp>

namespace neu_check_tool {
	decltype(auto) check_range_close(neu::gpu_vector const& gv, neu::cpu_vector const& cv, double tol) {
		const int size = cv.size();
		BOOST_CHECK(static_cast<int>(gv.size()) == size);
		neu::cpu_vector cgv(size);
		boost::compute::copy(gv.begin(), gv.end(), cgv.begin());
		for(int i = 0; i < size; ++i) {
			BOOST_CHECK_CLOSE(cgv[i], cv[i], tol);
		}
	}
}

#endif // NEU_TEST_CHECK_TOOL_HPP
