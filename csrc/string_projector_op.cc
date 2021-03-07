#include "bunruija/string_projector_op.h"


namespace bunruija {


StringProjectorOp::StringProjectorOp(int feature_size)
  : hasher_(Hasher(feature_size)) {
}

bool StringProjectorOp::is_training() {
  return false;
}

int StringProjectorOp::operator() (const std::string &word) {
  int code = hasher_(word);
  return code;
}
  
} // namespace bunruija
