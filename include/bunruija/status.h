#pragma once
#include <string>


namespace bunruija {


struct Status {
  int status_code;
  std::string status_message;
  Status() : status_code(0), status_message("") {}
  Status(int status_code, std::string status_message)
    : status_code(status_code), status_message(status_message) {}
};

#define RETURN_STATUS_IF_NOT_EQ(x, y, errmsg) \
  if (x == y) {                               \
  } else {                                    \
    return Status(x, errmsg);                 \
  }                                           \


#define ABORT_IF_NOT_EQ(x, y, errmsg)                                       \
  if (x == y) {                                                             \
  } else {                                                                  \
    std::cerr << __FILE__ << "(" << __LINE__ << ")" << errmsg << std::endl; \
    exit(x);                                                                \
  }                                                                         \


} // namespace bunruija
