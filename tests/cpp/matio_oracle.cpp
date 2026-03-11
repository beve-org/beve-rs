#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <matio.h>

namespace {

struct MatCloser {
  void operator()(mat_t* mat) const {
    if (mat != nullptr) {
      Mat_Close(mat);
    }
  }
};

struct MatVarCloser {
  void operator()(matvar_t* var) const {
    if (var != nullptr) {
      Mat_VarFree(var);
    }
  }
};

using MatFilePtr = std::unique_ptr<mat_t, MatCloser>;
using MatVarPtr = std::unique_ptr<matvar_t, MatVarCloser>;

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error(std::string(message));
}

size_t numel(const matvar_t* var) {
  if (var == nullptr) {
    return 0;
  }
  size_t total = 1;
  for (int idx = 0; idx < var->rank; ++idx) {
    total *= var->dims[idx];
  }
  return total;
}

void write_json_string(std::ostream& out, std::string_view value) {
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(ch) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  out << '"';
}

void append_utf8(std::string& out, uint32_t cp) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string decode_utf16(const mat_uint16_t* data, size_t len) {
  std::string out;
  out.reserve(len);
  for (size_t idx = 0; idx < len; ++idx) {
    const uint16_t unit = data[idx];
    if (unit >= 0xD800 && unit <= 0xDBFF && idx + 1 < len) {
      const uint16_t next = data[idx + 1];
      if (next >= 0xDC00 && next <= 0xDFFF) {
        const uint32_t high = static_cast<uint32_t>(unit - 0xD800);
        const uint32_t low = static_cast<uint32_t>(next - 0xDC00);
        append_utf8(out, 0x10000 + ((high << 10) | low));
        ++idx;
        continue;
      }
    }
    append_utf8(out, unit);
  }
  return out;
}

std::string decode_utf32(const mat_uint32_t* data, size_t len) {
  std::string out;
  out.reserve(len);
  for (size_t idx = 0; idx < len; ++idx) {
    append_utf8(out, data[idx]);
  }
  return out;
}

std::string decode_char_data(const matvar_t* var) {
  const size_t len = numel(var);
  if (len == 0 || var->data == nullptr) {
    return {};
  }

  switch (var->data_type) {
    case MAT_T_UTF8:
    case MAT_T_UINT8:
      return std::string(static_cast<const char*>(var->data), len);
    case MAT_T_UTF16:
    case MAT_T_UINT16:
      return decode_utf16(static_cast<const mat_uint16_t*>(var->data), len);
    case MAT_T_UTF32:
    case MAT_T_UINT32:
      return decode_utf32(static_cast<const mat_uint32_t*>(var->data), len);
    default:
      fail("unsupported MATIO char encoding");
  }
}

void write_dims(std::ostream& out, const matvar_t* var) {
  out << '[';
  for (int idx = 0; idx < var->rank; ++idx) {
    if (idx > 0) {
      out << ',';
    }
    out << var->dims[idx];
  }
  out << ']';
}

template <typename T>
void write_numeric_list(std::ostream& out, const T* data, size_t len) {
  out << '[';
  if constexpr (std::is_floating_point_v<T>) {
    out << std::setprecision(std::numeric_limits<T>::max_digits10);
  }
  for (size_t idx = 0; idx < len; ++idx) {
    if (idx > 0) {
      out << ',';
    }
    if constexpr (std::is_same_v<T, int8_t>) {
      out << static_cast<int>(data[idx]);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      out << static_cast<unsigned int>(data[idx]);
    } else {
      out << data[idx];
    }
  }
  out << ']';
}

void write_logical_list(std::ostream& out, const matvar_t* var) {
  const size_t len = numel(var);
  const auto* data = static_cast<const mat_uint8_t*>(var->data);
  out << '[';
  for (size_t idx = 0; idx < len; ++idx) {
    if (idx > 0) {
      out << ',';
    }
    out << ((data != nullptr && data[idx] != 0) ? "true" : "false");
  }
  out << ']';
}

template <typename T>
void write_numeric_value(std::ostream& out, const matvar_t* var, std::string_view kind) {
  out << "{\"kind\":";
  write_json_string(out, kind);
  out << ",\"dims\":";
  write_dims(out, var);
  out << ",\"data\":";
  const auto* data = static_cast<const T*>(var->data);
  if (numel(var) > 0 && data == nullptr) {
    fail("numeric array is missing data");
  }
  write_numeric_list(out, data, numel(var));
  out << '}';
}

template <typename T>
void write_complex_value(std::ostream& out, const matvar_t* var, std::string_view kind) {
  out << "{\"kind\":";
  write_json_string(out, kind);
  out << ",\"dims\":";
  write_dims(out, var);
  out << ",\"real\":";
  const auto* split = static_cast<const mat_complex_split_t*>(var->data);
  const auto* real = split == nullptr ? nullptr : static_cast<const T*>(split->Re);
  const auto* imag = split == nullptr ? nullptr : static_cast<const T*>(split->Im);
  if (numel(var) > 0 && (real == nullptr || imag == nullptr)) {
    fail("complex array is missing split data");
  }
  write_numeric_list(out, real, numel(var));
  out << ",\"imag\":";
  write_numeric_list(out, imag, numel(var));
  out << '}';
}

void write_value(std::ostream& out, const matvar_t* var);

void write_struct_value(std::ostream& out, const matvar_t* var) {
  out << "{\"kind\":\"struct\",\"dims\":";
  write_dims(out, var);
  out << ",\"fields\":{";

  const auto nfields = Mat_VarGetNumberOfFields(const_cast<matvar_t*>(var));
  const auto* names = Mat_VarGetStructFieldnames(var);
  const auto nelems = numel(var);
  for (unsigned field_idx = 0; field_idx < nfields; ++field_idx) {
    if (field_idx > 0) {
      out << ',';
    }
    const char* field_name = (names != nullptr) ? names[field_idx] : "";
    write_json_string(out, field_name == nullptr ? "" : field_name);
    out << ':';
    out << '[';
    for (size_t elem_idx = 0; elem_idx < nelems; ++elem_idx) {
      if (elem_idx > 0) {
        out << ',';
      }
      auto* field =
          Mat_VarGetStructFieldByIndex(const_cast<matvar_t*>(var), field_idx, elem_idx);
      if (field == nullptr) {
        fail("struct field is missing data");
      }
      write_value(out, field);
    }
    out << ']';
  }
  out << "}}";
}

void write_cell_value(std::ostream& out, const matvar_t* var) {
  out << "{\"kind\":\"cell\",\"dims\":";
  write_dims(out, var);
  out << ",\"elements\":";
  out << '[';
  const auto nelems = numel(var);
  for (size_t idx = 0; idx < nelems; ++idx) {
    if (idx > 0) {
      out << ',';
    }
    auto* cell = Mat_VarGetCell(const_cast<matvar_t*>(var), static_cast<int>(idx));
    if (cell == nullptr) {
      fail("cell element is missing data");
    }
    write_value(out, cell);
  }
  out << "]}";
}

void write_value(std::ostream& out, const matvar_t* var) {
  if (var == nullptr) {
    fail("null MAT variable");
  }

  if (var->class_type == 0 && var->data_type == 0) {
    out << "{\"kind\":\"unknown\",\"dims\":";
    write_dims(out, var);
    out << ",\"class_type\":0,\"data_type\":0}";
    return;
  }

  if (var->isLogical) {
    out << "{\"kind\":\"logical\",\"dims\":";
    write_dims(out, var);
    out << ",\"data\":";
    write_logical_list(out, var);
    out << '}';
    return;
  }

  switch (var->class_type) {
    case MAT_C_CHAR:
      out << "{\"kind\":\"char\",\"dims\":";
      write_dims(out, var);
      out << ",\"value\":";
      write_json_string(out, decode_char_data(var));
      out << '}';
      return;
    case MAT_C_DOUBLE:
      if (var->isComplex) {
        write_complex_value<double>(out, var, "complex_double");
      } else {
        write_numeric_value<double>(out, var, "double");
      }
      return;
    case MAT_C_SINGLE:
      if (var->isComplex) {
        write_complex_value<float>(out, var, "complex_single");
      } else {
        write_numeric_value<float>(out, var, "single");
      }
      return;
    case MAT_C_INT8:
      write_numeric_value<int8_t>(out, var, "int8");
      return;
    case MAT_C_UINT8:
      write_numeric_value<uint8_t>(out, var, "uint8");
      return;
    case MAT_C_INT16:
      write_numeric_value<int16_t>(out, var, "int16");
      return;
    case MAT_C_UINT16:
      write_numeric_value<uint16_t>(out, var, "uint16");
      return;
    case MAT_C_INT32:
      write_numeric_value<int32_t>(out, var, "int32");
      return;
    case MAT_C_UINT32:
      write_numeric_value<uint32_t>(out, var, "uint32");
      return;
    case MAT_C_INT64:
      write_numeric_value<int64_t>(out, var, "int64");
      return;
    case MAT_C_UINT64:
      write_numeric_value<uint64_t>(out, var, "uint64");
      return;
    case MAT_C_STRUCT:
      write_struct_value(out, var);
      return;
    case MAT_C_CELL:
      write_cell_value(out, var);
      return;
    case MAT_C_EMPTY:
      out << "{\"kind\":\"empty\",\"dims\":";
      write_dims(out, var);
      out << '}';
      return;
    default:
      fail("unsupported MATLAB class in MATIO oracle");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string_view(argv[1]) != "dump") {
      std::cerr << "usage: matio_oracle dump <path>\n";
      return 2;
    }

    MatFilePtr mat(Mat_Open(argv[2], MAT_ACC_RDONLY));
    if (!mat) {
      std::cerr << "failed to open MAT file\n";
      return 3;
    }

    std::cout << "{\"variables\":{";
    bool first = true;
    while (matvar_t* raw = Mat_VarReadNext(mat.get())) {
      MatVarPtr var(raw);
      if (var->name == nullptr || var->name[0] == '#') {
        continue;
      }
      if (!first) {
        std::cout << ',';
      }
      first = false;
      write_json_string(std::cout, var->name);
      std::cout << ':';
      write_value(std::cout, var.get());
    }
    std::cout << "}}\n";
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
