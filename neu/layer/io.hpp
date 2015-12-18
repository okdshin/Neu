#ifndef NEU_LAYER_IO_HPP
#define NEU_LAYER_IO_HPP
//20151219
#include <exception>
#include <iosfwd>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <neu/layer/traits.hpp>
#include <neu/layer/deserialize.hpp>
namespace neu {
	namespace layer {
		class file_open_error : public std::exception {
		public:
			file_open_error(std::string const& filename) : filename_(filename) {}

			const char* what() const noexcept override {
				return ("cannot open the file\""+filename_+"\"").c_str();
			}

		private:
			std::string filename_;
		};

		template<typename Layer>
		decltype(auto) output(Layer const& l, std::ostream& os,
				boost::compute::command_queue& queue) {
			YAML::Emitter emitter;
			::neu::layer::serialize(l, emitter, queue);
			os << emitter.c_str();
		}
		template<typename Layer>
		decltype(auto) output_to_file(Layer const& l, std::string const& filename,
				boost::compute::command_queue& queue) {
			std::ofstream ofs(filename.c_str());
			if(!ofs) { throw file_open_error(filename); }
			::neu::layer::output(l, ofs, queue);
		}

		decltype(auto) input(std::istream& is,
				boost::compute::command_queue& queue) {
			return deserialize(YAML::Load(is), queue);
		}
		decltype(auto) input_from_file(std::string const& filename,
				boost::compute::command_queue& queue) {
			std::ifstream ifs(filename.c_str());
			if(!ifs) { throw file_open_error(filename); }
			return input(ifs, queue);
		}
		
	}
}// namespace neu

#endif //NEU_LAYER_IO_HPP
