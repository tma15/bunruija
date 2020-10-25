#include <fstream>
#include <iostream>
#include <sqlite3.h>
#include <vector>

#include "bunruija_vector.h"


namespace bunruija {

uint32_t decode_utf8(const char *begin, const char *end, size_t *mblen) {
  const size_t len = end - begin;
//  std::cout << "begin[0]" << begin[0] << std::endl;
//  std::cout << "len" << len << std::endl;
//  std::cout << "uc" << static_cast<unsigned char>(begin[0]) << std::endl;

  if (static_cast<unsigned char>(begin[0]) < 0x80) {
    *mblen = 1;
//    std::cout << "mblen:1" << std::endl;
    return static_cast<unsigned char>(begin[0]);
  } else if (len >= 2 && (begin[0] & 0xE0) == 0xC0) {
//    std::cout << "mblen:2" << std::endl;
    const uint32_t cp = (((begin[0] & 0x1F) << 6) | ((begin[1] & 0x3F)));
//    if (IsTrailByte(begin[1]) && cp >= 0x0080 && IsValidCodepoint(cp)) {
    if (cp >= 0x0080) {
      *mblen = 2;
      return cp;
    }
  } else if (len >= 3 && (begin[0] & 0xF0) == 0xE0) {
//    std::cout << "mblen:3" << std::endl;
    const uint32_t cp = (((begin[0] & 0x0F) << 12) | ((begin[1] & 0x3F) << 6) |
                       ((begin[2] & 0x3F)));
    if (cp >= 0x0800) {
      *mblen = 3;
      return cp;
    }
  } else if (len >= 4 && (begin[0] & 0xf8) == 0xF0) {
//    std::cout << "mblen:4" << std::endl;
    const uint32_t cp = (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3F) << 12) |
                       ((begin[2] & 0x3F) << 6) | ((begin[3] & 0x3F)));
//    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) &&
//        IsTrailByte(begin[3]) && cp >= 0x10000 && IsValidCodepoint(cp)) {
    if (cp >= 0x10000) {
      *mblen = 4;
      return cp;
    }
  }

  // Invalid UTF-8.
  std::cout << "Invalid" << std::endl;
  *mblen = 1;
//  return kUnicodeError;
}


int PretrainedVectorProcessor::convert(const char *input_file) {
  std::cout << "test: " << input_file << std::endl;

  int ret = sqlite3_open_v2(
    "test.db", &db_, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, nullptr);
  
  std::ifstream ifs(input_file);
  std::string line;
  int k = 0;
  while (std::getline(ifs, line)) {
    process_line(&line);
    k++;
    if (k >= 2) {
      exit(1);
    }
  }
  return 0;
}


int PretrainedVectorProcessor::process_line(std::string *line) {
    std::cout << "#" << *line << std::endl;
    const char *begin = line->data();
    const char *end = line->data() + line->size();

    std::string word_delimiter(" ");
    std::string vector_delimiter(" ");

    std::string word;

    size_t word_mb_length = 0;

    while (true) {
      size_t multibyte_len;
      const uint32_t uc = decode_utf8(begin, end, &multibyte_len);

      std::string c(begin, begin + multibyte_len);

      begin += multibyte_len;

      if (c == word_delimiter) {
        std::string word(line->begin(), line->begin() + word_mb_length);
        std::cout << "word:" << word << std::endl;
        break;
      }
      word_mb_length += multibyte_len;
    }

    bool found_vector_delimiter = false;
    size_t number_length = 0;
    auto number_begin = line->begin() + word_mb_length + 1;
    while (begin < end) {
      std::string c(begin, begin + 1);
      ++begin;

      if (c == vector_delimiter) {
        std::string number(number_begin, number_begin + number_length);
        float val = std::stof(number);
        found_vector_delimiter = true;
        std::cout << "Num: \"" << val << "\"" << std::endl;
        number_begin = number_begin + number_length + 1;
        number_length = 0;
      } else {
        ++number_length;
      }
    }

    if (found_vector_delimiter) {
      exit(1);
    }
    return found_vector_delimiter;
}

  
} // namespace bunruija
