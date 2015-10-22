#ifndef NEU_LOCAL_RESPONCE_NORMALIZATION_LAYER_HPP
#define NEU_LOCAL_RESPONCE_NORMALIZATION_LAYER_HPP
//20150622
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
namespace neu {
	const char local_responce_normalization_across_maps_kernel_source[] =
	BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void local_responce_normalization_across_maps(
			const __global float* input, __global float* output, __global float* denoms,
			const int n, const float alpha, const float beta,
			const int channel_num, const int inoutput_width)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int k = 0; k < channel_num; ++k) {
				float squared_input_val_sum = 0;
				const int ch_start = max(0, k-n/2);
				const int ch_end = min(channel_num, k-n/2+n);
				for(int k_dash = ch_start; k_dash < ch_end; ++k_dash) {
					const int input_index = index(oc, or, k_dash, b,
						inoutput_width, channel_num);
					const float input_val = input[input_index];
					squared_input_val_sum += input_val*input_val;
				}
				const int inoutput_index = index(oc, or, k, b,
					inoutput_width, channel_num);
				const float denom = powr(1.f+(alpha/n)*squared_input_val_sum, beta);
				denoms[inoutput_index] = denom;
				output[inoutput_index] = input[inoutput_index] / denom;
			}
		}
	);

	const char local_responce_normalization_across_maps_back_kernel_source[] =
	BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void local_responce_normalization_across_maps_back(
			const __global float* input, __global float* output,
			const __global float* denoms,
			const int n, const float alpha, const float beta,
			const int channel_num, const int inoutput_width)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int k = 0; k < channel_num; ++k) {
				const int inoutput_index = index(oc, or, k, b,
					inoutput_width, channel_num);
				const float denom = denoms[inoutput_index];
				const float input_val = input[inoutput_index];
				output[inoutput_index] =
					(denom*n-2*alpha*beta*input_val*input_val)/(powr(denom, beta+1.f)*n);
			}
		}
	);

	const char local_responce_normalization_same_map_kernel_source[] =
	BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void local_responce_normalization_same_map(
			const __global float* input, __global float* output, __global float* denoms,
			const int n, const float alpha, const float beta,
			const int channel_num, const int inoutput_width)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			for(int k = 0; k < channel_num; ++k) {
				float squared_input_val_sum = 0;
				const int fr_start = max(0, or-n/2);
				const int fr_end = min(inoutput_width, or-n/2+n);
				for(int fr = fr_start; fr < fr_end; ++fr) {
					const int fc_start = max(0, oc-n/2);
					const int fc_end = min(inoutput_width, oc-n/2+n);
					for(int fc = fc_start; fc < fc_end; ++fc) {
						const int input_index = index(fc, fr, k, b,
							inoutput_width, channel_num);
						const float input_val = input[input_index];
						squared_input_val_sum += input_val*input_val;
					}
				}
				const int inoutput_index = index(oc, or, k, b,
					inoutput_width, channel_num);
				const float denom = powr(1.f+(alpha/(n*n))*squared_input_val_sum, beta);
				denoms[inoutput_index] = denom;
				output[inoutput_index] = input[inoutput_index] / denom;
			}
		}
	);

	const char local_responce_normalization_same_map_back_kernel_source[] =
	BOOST_COMPUTE_STRINGIZE_SOURCE(
		int index(int x, int y, int c, int id, int width, int channel_num) {
			return id*channel_num*width*width +c*width*width +y*width +x; }
		__kernel void local_responce_normalization_same_map_back(
			const __global float* input, __global float* output,
			const __global float* denoms,
			const int n, const float alpha, const float beta,
			const int channel_num, const int inoutput_width)
		{
			const int b = get_global_id(2);
			const int or = get_global_id(1);
			const int oc = get_global_id(0);

			const int squared_n = n*n;
			for(int k = 0; k < channel_num; ++k) {
				const int inoutput_index = index(oc, or, k, b,
					inoutput_width, channel_num);
				const float denom = denoms[inoutput_index];
				const float input_val = input[inoutput_index];
				output[inoutput_index] =
					(denom*squared_n-2*alpha*beta*input_val*input_val)/(powr(denom, beta+1)*squared_n);
			}
		}
	);

	class local_responce_normalization_layer {
	public:
		local_responce_normalization_layer(
			std::size_t n, float alpha, float beta,
			std::size_t inoutput_width, std::size_t channel_num, std::size_t batch_size,
			boost::compute::kernel const& normalization_kernel,
			boost::compute::kernel const& normalization_back_kernel)
			: n_(n), alpha_(alpha), beta_(beta),
			inoutput_width_(inoutput_width),
			channel_num_(channel_num), batch_size_(batch_size),
			normalization_kernel_(normalization_kernel),
			normalization_back_kernel_(normalization_back_kernel),
			next_input_(inoutput_width_*inoutput_width_*channel_num_*batch_size_),
			denom_(inoutput_width_*inoutput_width_*channel_num_*batch_size_),
			prev_delta_(inoutput_width_*inoutput_width_*channel_num_*batch_size_) {}

		decltype(auto) forward(neu::gpu_vector const& input) {
			execute_nd_range_kernel<3>(normalization_kernel_,
				{0, 0, 0}, {inoutput_width_, inoutput_width_, batch_size_},
				input, next_input_, denom_,
				static_cast<int>(n_), alpha_, beta_,
				static_cast<int>(channel_num_), static_cast<int>(inoutput_width_));
		}
		decltype(auto) get_next_input() const { return (next_input_); }

		decltype(auto) backward(neu::gpu_vector const& delta) {
			execute_nd_range_kernel<3>(normalization_back_kernel_,
				{0, 0, 0}, {inoutput_width_, inoutput_width_, batch_size_},
				delta, prev_delta_, denom_,
				static_cast<int>(n_), alpha_, beta_,
				static_cast<int>(channel_num_), static_cast<int>(inoutput_width_));
		}
		decltype(auto) get_prev_delta() const { return (prev_delta_); }

		decltype(auto) update() { /* do nothing */ }
	private:
		const std::size_t n_;
		const float alpha_;
		const float beta_;

		const std::size_t inoutput_width_;
		const std::size_t channel_num_;
		const std::size_t batch_size_;

		boost::compute::kernel normalization_kernel_;
		boost::compute::kernel normalization_back_kernel_;

		neu::gpu_vector next_input_;
		neu::gpu_vector denom_;
		neu::gpu_vector prev_delta_;
	};
	decltype(auto) make_local_responce_normalization_across_maps_layer(
		std::size_t n, float alpha, float beta,
		std::size_t inoutput_width, std::size_t channel_num, std::size_t batch_size
	) {
		auto across_maps_kernel = make_kernel(
			local_responce_normalization_across_maps_kernel_source,
			"local_responce_normalization_across_maps");
		auto across_maps_back_kernel = make_kernel(
			local_responce_normalization_across_maps_back_kernel_source,
			"local_responce_normalization_across_maps_back");
		return local_responce_normalization_layer(n, alpha, beta,
			inoutput_width, channel_num, batch_size,
			across_maps_kernel, across_maps_back_kernel);
	}
	decltype(auto) make_local_responce_normalization_same_map_layer(
		std::size_t n, float alpha, float beta,
		std::size_t inoutput_width, std::size_t channel_num, std::size_t batch_size
	) {
		auto same_map_kernel = make_kernel(
			local_responce_normalization_same_map_kernel_source,
			"local_responce_normalization_same_map");
		auto same_map_back_kernel = make_kernel(
			local_responce_normalization_same_map_back_kernel_source,
			"local_responce_normalization_same_map_back");
		return local_responce_normalization_layer(n, alpha, beta,
			inoutput_width, channel_num, batch_size,
			same_map_kernel, same_map_back_kernel);
	}
}// namespace neu

#endif //NEU_LOCAL_RESPONCE_NORMALIZATION_LAYER_HPP
