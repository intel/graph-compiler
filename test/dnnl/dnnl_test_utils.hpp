#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_str_resource(const std::string &name) {
  std::filesystem::path res_dir{"resources"};
  auto path = std::filesystem::absolute(res_dir / name);
  std::ifstream file(path);

  if (!file) {
    throw std::runtime_error("Unable to open file " + path.string());
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  return buffer.str();
}
