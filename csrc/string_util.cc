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


void Hasher::get_hash_codes(const std::string &word, std::vector<uint64_t> *hash_codes) {
  hash_codes->clear();

  uint64_t hash_high = 0;
  uint64_t hash_low = 0;
  for (int i = 0; i < feature_size_; i += 64) {
    if (i == 0) {
      std::pair<uint64_t, uint64_t> hash = murmur_hash(word.c_str(), word.size());
      hash_low = hash.first;
      hash_high = hash.second;
    } else {
      get_more_bits(hash_low, hash_high, &hash_low, &hash_high);
    }
//    std::cout << "low/high:" << hash_low << " " << hash_high << std::endl;
    hash_codes->push_back(hash_low);
    hash_codes->push_back(hash_high);
  }
}


std::pair<uint64_t, uint64_t> Hasher::murmur_hash(const char *data, const size_t len) {
  uint64_t hash = len * kMul;
  uint64_t hash2 = 0;
  const size_t len_aligned = len & ~0x7;
  const char *end = data + len_aligned;

//  std::cout << std::endl;
//  std::cout << "len_aligned:" << len_aligned << " " << len << std::endl;
//  std::cout << "data:" << data << " end:" << end << std::endl;
  for (const char *p = data; p != end; p += 8) {
    hash = step(hash, load_64variable_length(p, 8));
    hash2 ^= hash;
//    std::cout << "hash:" << hash << " hash2:" << hash2 << std::endl;
  }

  if ((len & 0x7) != 0) {
    const uint64_t data = load_64variable_length(end, len & 0x7);
    hash ^= data;
    hash *= kMul;
    hash2 ^= hash;
  }
  hash = shift_mix(hash) * kMul;
  hash2 ^= hash;
  hash = shift_mix(hash);
  hash2 = shift_mix(hash2 * kMul2) * kMul2;
  return std::make_pair(hash, hash2);
}


uint64_t Hasher::load_64variable_length(const void *p, int len) {
  assert(len >= 1 && len <= 8);
  const char* buf = static_cast<const char*>(p);
  uint64_t val = 0;
  --len;
  do {
    val = (val << 8) | buf[len];
  } while (--len >= 0);
  return val;
}


uint64_t Hasher::step(uint64_t hash, int data) {
  hash ^= shift_mix(data * kMul) * kMul;
  hash *= kMul;
  return hash;
}


uint64_t Hasher::shift_mix(uint64_t x) {
  return x ^ (x >> 47);
}


void Hasher::get_more_bits(uint64_t hash, uint64_t hash2, uint64_t *rlow, uint64_t *rhigh) {
  hash = shift_mix(hash) * kMul;
  hash2 ^= hash;
  *rhigh = shift_mix(hash);
  *rlow = shift_mix(hash2 * kMul2) * kMul2;
}


} // namespace bunruija
