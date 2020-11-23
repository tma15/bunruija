#pragma once
#include <sqlite3.h>


namespace bunruija {

void one_char_length(const char *begin, const char *end, size_t *mblen);


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


class PretrainedVectorProcessor {
  public:
    PretrainedVectorProcessor(): length_(0), precision_(7) {}

    Status convert(const char *db_file, const char *input_file);
    Status query(const std::string &query, std::vector<float> *vector);
    Status load(const std::string &input_file);
    Status contains(const std::string &query, bool *has);
    int dim() { return dim_; }
    int length() { return length_; }

  private:
    Status initialize_db(const char *db_file, std::string *line);
    Status process_line(std::string *line);

    sqlite3 *db_;
    sqlite3_stmt *stmt_;

    int dim_;
    int length_;
    int precision_;
    bool is_word2vec_format_;
};

} // bunruija
