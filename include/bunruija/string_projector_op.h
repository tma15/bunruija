#pragma once
#include "bunruija/status.h"
#include "internal/string_util.h"


namespace bunruija {


class StringProjectorOp {
  public:
    StringProjectorOp() {}
    StringProjectorOp(int feature_size);
    bool is_training();
    int operator()(const std::string &word);

  private:
    Hasher hasher_;
    bool is_training_;

};
  
} // namespace bunruija
