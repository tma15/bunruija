#pragma once
#include <vector>

#include "bunruija/status.h"
#include "internal/string_util.h"


namespace bunruija {

constexpr float kmapping_table[4] = {0, 1, -1, 0};
constexpr int kincrement = 32;


class StringProjectorOp {
  public:
    StringProjectorOp() {}
    StringProjectorOp(int feature_size);
    bool is_training();
    void operator()(const std::vector<std::vector<std::string>> &words, float *projection);

  private:
    Hasher hasher_;
    int feature_size_;
    bool is_training_;

};
  
} // namespace bunruija
