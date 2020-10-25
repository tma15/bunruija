#pragma once
#include <sqlite3.h>


namespace bunruija {

uint32_t decode_utf8(const char *begin, const char *end, size_t *mblen);


class PretrainedVectorProcessor {
  public:
    PretrainedVectorProcessor() {}

    int convert(const char *input_file);

  private:
    int process_line(std::string *line);

    sqlite3 *db_;
};

} // bunruija
