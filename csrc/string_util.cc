#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "internal/string_util.h"


namespace bunruija {

void one_char_length(const char *begin, const char *end, size_t *mblen) {
  const size_t len = end - begin;

  if (static_cast<unsigned char>(begin[0]) < 0x80) {
    *mblen = 1;
  } else if (len >= 2 && (begin[0] & 0xE0) == 0xC0) {
    const uint32_t cp = (((begin[0] & 0x1F) << 6) | ((begin[1] & 0x3F)));
    if (cp >= 0x0080) {
      *mblen = 2;
    }
  } else if (len >= 3 && (begin[0] & 0xF0) == 0xE0) {
    const uint32_t cp = (((begin[0] & 0x0F) << 12) | ((begin[1] & 0x3F) << 6) |
                       ((begin[2] & 0x3F)));
    if (cp >= 0x0800) {
      *mblen = 3;
    }
  } else if (len >= 4 && (begin[0] & 0xf8) == 0xF0) {
    const uint32_t cp = (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3F) << 12) |
                       ((begin[2] & 0x3F) << 6) | ((begin[3] & 0x3F)));
    if (cp >= 0x10000) {
      *mblen = 4;
    }
  }
  // Invalid UTF-8.
  *mblen = 1;
}


int Hasher::operator()(const std::string &word) {
  for (int i = 0; i < feature_size_; i += 64) {
    if (i == 0) {
      std::pair<int, int> hash = murmur_hash(word.c_str(), word.size());
    } else {
    }
  }
  return 1;
}


std::pair<int, int> Hasher::murmur_hash(const char *data, const size_t len) {
  std::cout << "murmur" << std::endl;
  return std::make_pair(1, 1);
}

  
} // namespace bunruija
