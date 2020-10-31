#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <sqlite3.h>
#include <vector>

#include "bunruija_vector.h"


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


Status PretrainedVectorProcessor::convert(const char *input_file) {
  std::cout << "test: " << input_file << std::endl;
  std::ifstream ifs(input_file);
  std::string line;
  int k = 0;
  int ret;
  while (std::getline(ifs, line)) {
    if (k == 0) {
      initialize_db(&line);

      std::stringstream query_ss;
      query_ss  << "INSERT INTO `bunruija`(key";
      for (int i = 0; i < dim_; ++i) {
        query_ss << ", dim_" << i;
      }
      query_ss << ") VALUES (?";
      for (int i = 0; i < dim_; ++i) {
        query_ss << ", ?";
      }
      query_ss << ");";
      std::string query = query_ss.str();

      ret = sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt_, nullptr);
      RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

      ret = sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, nullptr);
      RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
    }

    Status status = process_line(&line);
    if (status.status_code != 0) {
      std::cerr << status.status_message << std::endl;
      exit(status.status_code);
    }

    k++;
    if (k % 100000 == 0) {
      std::cerr << "Dumped " << k << " words" << std::endl;;
      ret = sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);
      RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
      break;
    }
  }

  RETURN_STATUS_IF_NOT_EQ(sqlite3_finalize(stmt_), SQLITE_OK, sqlite3_errmsg(db_))
  return Status(0, "");
}


Status PretrainedVectorProcessor::initialize_db(std::string *line) {
  const char *begin = line->data();
  const char *end = line->data() + line->size();

  std::string word_delimiter(" ");
  std::string vector_delimiter(" ");

  std::string word;

  size_t word_mb_length = 0;

  // Consume the first element
  while (true) {
    size_t multibyte_len;
    one_char_length(begin, end, &multibyte_len);
    std::string c(begin, begin + multibyte_len);

    begin += multibyte_len;

    if (c == word_delimiter) {
      std::string word(line->begin(), line->begin() + word_mb_length);
      break;
    }
    word_mb_length += multibyte_len;
  }

  // Infer dimension of word embeddings
  bool found_vector_delimiter = false;
  size_t number_length = 0;
  auto number_begin = line->begin() + word_mb_length + 1;
  std::vector<std::string> elems;
  while (begin < end) {
    std::string c(begin, begin + 1);
    ++begin;

    if (c == vector_delimiter) {
      std::string number(number_begin, number_begin + number_length);
      found_vector_delimiter = true;
      number_begin = number_begin + number_length + 1;
      number_length = 0;
      elems.push_back(number);
    } else {
      ++number_length;
    }
  }
  std::string number(number_begin, number_begin + number_length);

  if (found_vector_delimiter) {
    dim_ = elems.size();
  } else {
    dim_ = std::stoi(number);
  }

  int ret;

  ret = sqlite3_open_v2(
    "test.db",
    &db_, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE,
    nullptr);

  ret = sqlite3_exec(db_, "DROP TABLE IF EXISTS `bunruija`;", nullptr, nullptr, nullptr);

  std::stringstream ss;
  ss << "CREATE TABLE `bunruija` (key TEXT COLLATE NOCASE";
  for (int i = 0; i < dim_; ++i) {
    ss << ", dim_" << i << " INTEGER";
  }
  ss << ");";
  std::string create_stmt = ss.str();

//  std::cout << create_stmt << std::endl;

  ret = sqlite3_exec(
      db_,
      create_stmt.c_str(),
      nullptr,
      nullptr,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  std::stringstream ss_fmt;
  ss_fmt << "CREATE TABLE `bunruija_format` (key TEXT COLLATE NOCASE, value INTEGER);";
  std::string create_fmt_stmt = ss_fmt.str();

//  std::cout << create_fmt_stmt << std::endl;

  ret = sqlite3_exec(
      db_,
      create_fmt_stmt.c_str(),
      nullptr,
      nullptr,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  // Insert meta information
  std::string insert_query = "INSERT INTO `bunruija_format`(key, value) VALUES (?, ?);";
  ret = sqlite3_prepare_v2(db_, insert_query.c_str(), -1, &stmt_, nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  ret = sqlite3_bind_text(stmt_, 1, "dim", strlen("dim"), SQLITE_TRANSIENT);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  ret = sqlite3_bind_int64(stmt_, 2, dim_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  while (sqlite3_step(stmt_) == SQLITE_BUSY) {}
  ret = sqlite3_reset(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  ret = sqlite3_clear_bindings(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  ret = sqlite3_bind_text(stmt_, 1, "precision", strlen("precision"), SQLITE_TRANSIENT);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  ret = sqlite3_bind_int64(stmt_, 2, precision_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  while (sqlite3_step(stmt_) == SQLITE_BUSY) {}
  ret = sqlite3_reset(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  ret = sqlite3_clear_bindings(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  return Status(0, "");
}


Status PretrainedVectorProcessor::process_line(std::string *line) {
//  std::cout << "#" << *line << std::endl;
  const char *begin = line->data();
  const char *end = line->data() + line->size();


  std::string word_delimiter(" ");
  std::string vector_delimiter(" ");

  std::string word;

  int ret;

  size_t word_mb_length = 0;

  while (true) {
    size_t multibyte_len;
    one_char_length(begin, end, &multibyte_len);

    std::string c(begin, begin + multibyte_len);

    begin += multibyte_len;

    if (c == word_delimiter) {
      std::string word(line->begin(), line->begin() + word_mb_length);
//      std::cout << "word:" << word << std::endl;

      ret = sqlite3_bind_text(stmt_, 1, word.c_str(), strlen(word.c_str()), SQLITE_TRANSIENT);
      RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

      break;
    }
    word_mb_length += multibyte_len;
  }

  size_t number_length = 0;
  auto number_begin = line->begin() + word_mb_length + 1;
  unsigned d = 2;
  while (begin < end) {
    std::string c(begin, begin + 1);
    ++begin;

    if (c == vector_delimiter) {
      std::string number(number_begin, number_begin + number_length);
      float val = std::stof(number);
      int v = val * pow(10, precision_);
//      std::cout << "Num" << d << ": \"" << v << "\"" << std::endl;

      ret = sqlite3_bind_int64(stmt_, d, v);
      RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
      ++d;

      number_begin = number_begin + number_length + 1;
      number_length = 0;
    } else {
      ++number_length;
    }
  }
  std::string number(number_begin, number_begin + number_length);
  float val = std::stof(number);
  int v = val * pow(10, precision_);
//  std::cout << "Last Num" << d << ": \"" << v << "\"" << std::endl;

  ret = sqlite3_bind_int64(stmt_, d, v);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  while (sqlite3_step(stmt_) == SQLITE_BUSY) {}
  ret = sqlite3_reset(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  ret = sqlite3_clear_bindings(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  return Status(0, "");
}


Status PretrainedVectorProcessor::query(const std::string &query, std::vector<float> *vector) {
  return Status(0, "");
}


Status PretrainedVectorProcessor::load(const std::string &input_file) {
  return Status(0, "");
}

} // namespace bunruija
