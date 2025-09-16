#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <iostream>

#include "glaze/api/impl.hpp"
#include "glaze/beve/read.hpp"
#include "glaze/beve/write.hpp"
#include "glaze/beve/beve_to_json.hpp"

// Sample enum
enum class Color { Red, Green, Blue };

template <> struct glz::meta<Color> {
  static constexpr std::string_view name = "Color";
  static constexpr auto value = enumerate(
      "Red", Color::Red,
      "Green", Color::Green,
      "Blue", Color::Blue);
};

// Sample struct
struct Basic {
  int i{};
  double d{};
  std::string name{};
  std::vector<double> vs{};
  std::vector<bool> vb{};
  std::map<std::string,int> m{};
  std::map<int,double> mi{};
  Color color{Color::Green};
};

template <> struct glz::meta<Basic> {
  using T = Basic;
  static constexpr std::string_view name = "Basic";
  static constexpr auto value = object(
      "i", &T::i,
      "d", &T::d,
      "name", &T::name,
      "vs", &T::vs,
      "vb", &T::vb,
      "m", &T::m,
      "mi", &T::mi,
      "color", &T::color
  );
};

// Sample values
static auto sample_vec_f64() { return std::vector<double>{1.1, -2.25, 3.5}; }
static auto sample_vec_u32() { return std::vector<uint32_t>{1, 2, 1000}; }
static auto sample_vec_bool() { return std::vector<bool>{true,false,true,true,false}; }
static auto sample_vec_string() { return std::vector<std::string>{"a","bb","ccc"}; }
static auto sample_color() { return Color::Green; }
static auto sample_basic() {
  Basic b{};
  b.i = 42; b.d = -3.125; b.name = "hello";
  b.vs = sample_vec_f64();
  b.vb = sample_vec_bool();
  b.m = {{"a",1},{"bb",2}};
  b.mi = {{5, 3.14},{7, 7.42}};
  b.color = Color::Green;
  return b;
}

static bool eq(const Basic& a, const Basic& b) {
  return a.i==b.i && a.d==b.d && a.name==b.name && a.vs==b.vs && a.vb==b.vb && a.m==b.m && a.mi==b.mi && a.color==b.color;
}

static std::string read_stdin_all() {
  std::string s; std::string chunk;
  char buf[4096];
  while (true) {
    std::cin.read(buf, sizeof(buf));
    std::streamsize n = std::cin.gcount();
    if (n > 0) s.append(buf, buf + n);
    if (n < (std::streamsize)sizeof(buf)) break;
  }
  return s;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: glaze_interop <write|read> <case>\n";
    return 2;
  }
  std::string mode = argv[1];
  std::string cas = argv[2];

  if (mode == "write") {
    std::string out;
    if (cas == "vec_f64") { auto v=sample_vec_f64(); glz::write_beve(v, out); }
    else if (cas == "vec_u32") { auto v=sample_vec_u32(); glz::write_beve(v, out); }
    else if (cas == "vec_bool") { auto v=sample_vec_bool(); glz::write_beve(v, out); }
    else if (cas == "vec_string") { auto v=sample_vec_string(); glz::write_beve(v, out); }
    else if (cas == "color") { auto v=sample_color(); glz::write_beve(v, out); }
    else if (cas == "basic") { auto v=sample_basic(); glz::write_beve(v, out); }
    else if (cas == "cplx64") { std::complex<double> v{1.5, -2.25}; glz::write_beve(v, out); }
    else if (cas == "vcplx64") { std::vector<std::complex<double>> v{{1.0,2.0},{-3.0,4.5}}; glz::write_beve(v, out); }
    else if (cas == "mapi8") { std::map<int8_t,int> m{{-1,10},{2,20}}; glz::write_beve(m, out); }
    else if (cas == "mapu16") { std::map<uint16_t,int> m{{1,10},{65535,20}}; glz::write_beve(m, out); }
    else if (cas == "mapu64") { std::map<uint64_t,int> m{{1,10},{(uint64_t)1<<40,20}}; glz::write_beve(m, out); }
    else { std::cerr << "unknown case\n"; return 3; }
    std::cout.write(out.data(), (std::streamsize)out.size());
    return 0;
  }
  else if (mode == "read") {
    auto bytes = read_stdin_all();
    if (cas == "vec_f64") { auto expect=sample_vec_f64(); decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "vec_u32") { auto expect=sample_vec_u32(); decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "vec_bool") { auto expect=sample_vec_bool(); decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "vec_string") { auto expect=sample_vec_string(); decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "color") { auto expect=sample_color(); decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "basic") { auto expect=sample_basic(); decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (eq(got, expect)) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "cplx64") { std::complex<double> expect{1.5,-2.25}; decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "vcplx64") { std::vector<std::complex<double>> expect{{1.0,2.0},{-3.0,4.5}}; decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "mapi8") { std::map<int8_t,int> expect{{-1,10},{2,20}}; decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "mapu16") { std::map<uint16_t,int> expect{{1,10},{65535,20}}; decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else if (cas == "mapu64") { std::map<uint64_t,int> expect{{1,10},{(uint64_t)1<<40,20}}; decltype(expect) got; if (glz::read_beve(got, bytes)) { std::cerr<<"parse error\n"; return 4; } if (got==expect) { std::cout<<"OK\n"; return 0; } }
    else { std::cerr << "unknown case\n"; return 3; }
    std::cout<<"MISMATCH\n"; return 5;
  }
  else if (mode == "tojson") {
    auto bytes = read_stdin_all();
    std::string json;
    auto ec = glz::beve_to_json(bytes, json);
    if (bool(ec)) { std::cerr << "tojson error\n"; return 6; }
    std::cout << json;
    return 0;
  }
  else {
    std::cerr << "unknown mode\n";
    return 2;
  }
}
