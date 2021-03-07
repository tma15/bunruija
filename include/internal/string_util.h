#pragma once
#include <string>


namespace bunruija {


void one_char_length(const char *begin, const char *end, size_t *mblen);


class Hasher {
  public:
    Hasher() {}
    Hasher(int feature_size) : feature_size_(feature_size) {};
    int operator()(const std::string &word);

  private:
    std::pair<int, int> murmur_hash(const char *data, const size_t len);
    int feature_size_;
};


} // namespace bunruija
