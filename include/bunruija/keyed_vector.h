#pragma once
#include <sqlite3.h>

#include "bunruija/status.h"


namespace bunruija {


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
