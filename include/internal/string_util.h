#pragma once
#include <string>
#include <vector>


namespace bunruija {


void one_char_length(const char *begin, const char *end, size_t *mblen);


class Hasher {
  public:
    Hasher() {}
    Hasher(int feature_size) : feature_size_(feature_size) {};
    void get_hash_codes(const std::string &word, std::vector<uint64_t> *hash_codes);

  private:
    std::pair<uint64_t, uint64_t> murmur_hash(const char *data, const size_t len);
    uint64_t step(uint64_t hash, int data);
    uint64_t shift_mix(uint64_t x);
    uint64_t load_64variable_length(const void *p, int len);
    void get_more_bits(uint64_t hash, uint64_t hash2, uint64_t *rlow, uint64_t *rhigh);
    static constexpr uint64_t kMul = 0xc6a4a7935bd1e995ULL;
    static constexpr uint64_t kMul2 = 0x9e3779b97f4a7835ULL;
    int feature_size_;
};


class StringDistorter {
  public:
    StringDistorter() {}
    StringDistorter(float distortion_probability);
    void distort(const std::string &string, std::string *output);

  private:
    float distortion_probability_;
};


} // namespace bunruija
