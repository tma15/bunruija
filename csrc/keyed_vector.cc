#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <sqlite3.h>
#include <unordered_map>
#include <vector>

#include "bunruija/keyed_vector.h"
#include "internal/string_util.h"


namespace bunruija {


Status PretrainedVectorProcessor::convert(const char *db_file, const char *input_file) {
  std::ifstream ifs(input_file);
  std::string line;
  int k = 0;
  int ret;
  while (std::getline(ifs, line)) {
    if (k == 0) {
      initialize_db(db_file, &line);

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

      RETURN_STATUS_IF_NOT_EQ(
        sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, nullptr),
        SQLITE_OK,
        sqlite3_errmsg(db_))

      if (is_word2vec_format_) {
        continue;
      }
    }

    Status status = process_line(&line);

    if (status.status_code != 0) {
      std::cerr << status.status_message << std::endl;
      exit(status.status_code);
    }

    k++;
    if (k % 10000 == 0) {
      std::cerr << "Dumped " << k << " words" << std::endl;;
      ret = sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);
      RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

      RETURN_STATUS_IF_NOT_EQ(
        sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, nullptr),
        SQLITE_OK,
        sqlite3_errmsg(db_))
    }
  }
  ret = sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  ret = sqlite3_exec(
      db_,
      "CREATE INDEX `bunruija_key_idx` ON `bunruija` (key);",
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

  ret = sqlite3_bind_text(stmt_, 1, "length", strlen("length"), SQLITE_TRANSIENT);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  ret = sqlite3_bind_int64(stmt_, 2, length_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  while (sqlite3_step(stmt_) == SQLITE_BUSY) {}
  ret = sqlite3_reset(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  ret = sqlite3_clear_bindings(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))


  ret = sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  RETURN_STATUS_IF_NOT_EQ(sqlite3_finalize(stmt_), SQLITE_OK, sqlite3_errmsg(db_))
  RETURN_STATUS_IF_NOT_EQ(sqlite3_close(db_), SQLITE_OK, sqlite3_errmsg(db_))
  return Status(0, "");
}


Status PretrainedVectorProcessor::initialize_db(const char *db_file, std::string *line) {
  const char *begin = line->data();
  const char *end = line->data() + line->size();

  std::string word_delimiter(" ");
  std::string vector_delimiter(" ");

  std::string word;

  size_t word_mb_length = 0;

  // Consume the first element
  while (true) {
//    size_t multibyte_len;
//    one_char_length(begin, end, &multibyte_len);
    size_t multibyte_len = one_char_length(begin);
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
  elems.push_back(number);

  if (found_vector_delimiter) {
    dim_ = elems.size();
    is_word2vec_format_ = false;
  } else {
    dim_ = std::stoi(number);
    is_word2vec_format_ = true;
  }

  int ret;

  ret = sqlite3_open_v2(
    db_file,
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

  ret = sqlite3_exec(
      db_,
      create_fmt_stmt.c_str(),
      nullptr,
      nullptr,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  return Status(0, "");
}


Status PretrainedVectorProcessor::process_line(std::string *line) {
  const char *begin = line->data();
  const char *end = line->data() + line->size();

  std::string word_delimiter(" ");
  std::string vector_delimiter(" ");

  std::string word;

  int ret;

  size_t word_mb_length = 0;

  while (true) {
//    size_t multibyte_len;
//    one_char_length(begin, end, &multibyte_len);
    size_t multibyte_len = one_char_length(begin);
    std::string c(begin, begin + multibyte_len);

    begin += multibyte_len;

    if (c == word_delimiter) {
      std::string word(line->begin(), line->begin() + word_mb_length);

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

  ret = sqlite3_bind_int64(stmt_, d, v);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  while (sqlite3_step(stmt_) == SQLITE_BUSY) {}
  ret = sqlite3_reset(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  ret = sqlite3_clear_bindings(stmt_);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  ++length_;

  return Status(0, "");
}


Status PretrainedVectorProcessor::query(const std::string &query, std::vector<float> *vector) {
  int ret;

  std::stringstream ss;
  ss << "SELECT * FROM bunruija WHERE key='" << query << "'";
  std::string q = ss.str();

  ret = sqlite3_prepare_v2(
      db_,
      q.c_str(),
      -1,
      &stmt_,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  while(sqlite3_step(stmt_) == SQLITE_ROW) {

    for (int d = 0; d < dim_; ++d) {
      int v = sqlite3_column_int64(stmt_, d + 1);
      float value = float(v) / float(pow(10, precision_));
      vector->push_back(value);
    }
    break;
  }
  RETURN_STATUS_IF_NOT_EQ(sqlite3_reset(stmt_), SQLITE_OK, sqlite3_errmsg(db_))
  RETURN_STATUS_IF_NOT_EQ(sqlite3_clear_bindings(stmt_), SQLITE_OK, sqlite3_errmsg(db_))

  return Status(0, "");
}


Status PretrainedVectorProcessor::batch_query(const std::vector<std::string> &query, std::unordered_map<std::string, std::vector<float>> *vector) {
  int ret;

  std::stringstream ss;
  ss << "SELECT * FROM bunruija WHERE key IN (";
  for (int i = 0; i < query.size(); ++i) {
    ss << "'" << query[i] << "'";
    if (i < query.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")";
  std::string q = ss.str();
//  std::cout << q << std::endl;

  ret = sqlite3_prepare_v2(
      db_,
      q.c_str(),
      -1,
      &stmt_,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  int word_offset = 0;
  while(sqlite3_step(stmt_) == SQLITE_ROW) {
    int offset = 0;
    const unsigned char *word_c = sqlite3_column_text(stmt_, offset);
    std::string word = std::string(reinterpret_cast<const char *>(word_c));
    ++offset;

    (*vector)[word] = std::vector<float>(dim_, 0);
    for (int d = 0; d < dim_; ++d) {
      int v = sqlite3_column_int64(stmt_, d + 1);
      float value = float(v) / float(pow(10, precision_));
      (*vector)[word][d] = value;
    }
//    std::cout << word << ":";
//    for (auto i : (*vector)[word]) { std::cout << i << " "; }
//    std::cout << std::endl;
    offset += dim_;
    ++word_offset;
  }
  RETURN_STATUS_IF_NOT_EQ(sqlite3_reset(stmt_), SQLITE_OK, sqlite3_errmsg(db_))
  RETURN_STATUS_IF_NOT_EQ(sqlite3_clear_bindings(stmt_), SQLITE_OK, sqlite3_errmsg(db_))

  return Status(0, "");
}


Status PretrainedVectorProcessor::contains(const std::string &query, bool *out) {
  int ret;

  std::stringstream ss;
  ss << "SELECT * FROM bunruija WHERE key='" << query << "'";
  std::string q = ss.str();

  ret = sqlite3_prepare_v2(
      db_,
      q.c_str(),
      -1,
      &stmt_,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  *out = false;
  while(sqlite3_step(stmt_) == SQLITE_ROW) {
    *out = true;
  }
  RETURN_STATUS_IF_NOT_EQ(sqlite3_reset(stmt_), SQLITE_OK, sqlite3_errmsg(db_))
  RETURN_STATUS_IF_NOT_EQ(sqlite3_clear_bindings(stmt_), SQLITE_OK, sqlite3_errmsg(db_))

  return Status(0, "");
}



Status PretrainedVectorProcessor::load(const std::string &input_file) {
  int ret = sqlite3_open_v2(
    input_file.c_str(),
    &db_, SQLITE_OPEN_READONLY,
    nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))

  ret = sqlite3_prepare_v2(
      db_,
      "SELECT value FROM bunruija_format WHERE key='dim'",
      -1,
      &stmt_,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  while(sqlite3_step(stmt_) == SQLITE_ROW) {
    dim_ = sqlite3_column_int64(stmt_, 0);
  }
  RETURN_STATUS_IF_NOT_EQ(sqlite3_reset(stmt_), SQLITE_OK, sqlite3_errmsg(db_))
  RETURN_STATUS_IF_NOT_EQ(sqlite3_clear_bindings(stmt_), SQLITE_OK, sqlite3_errmsg(db_))

  ret = sqlite3_prepare_v2(
      db_,
      "SELECT value FROM bunruija_format WHERE key='length'",
      -1,
      &stmt_,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  while(sqlite3_step(stmt_) == SQLITE_ROW) {
    length_ = sqlite3_column_int64(stmt_, 0);
  }
  RETURN_STATUS_IF_NOT_EQ(sqlite3_reset(stmt_), SQLITE_OK, sqlite3_errmsg(db_))
  RETURN_STATUS_IF_NOT_EQ(sqlite3_clear_bindings(stmt_), SQLITE_OK, sqlite3_errmsg(db_))

  ret = sqlite3_prepare_v2(
      db_,
      "SELECT value FROM bunruija_format WHERE key='precision'",
      -1,
      &stmt_,
      nullptr);
  RETURN_STATUS_IF_NOT_EQ(ret, SQLITE_OK, sqlite3_errmsg(db_))
  while(sqlite3_step(stmt_) == SQLITE_ROW) {
    precision_ = sqlite3_column_int64(stmt_, 0);
  }
  RETURN_STATUS_IF_NOT_EQ(sqlite3_reset(stmt_), SQLITE_OK, sqlite3_errmsg(db_))
  RETURN_STATUS_IF_NOT_EQ(sqlite3_clear_bindings(stmt_), SQLITE_OK, sqlite3_errmsg(db_))

  return Status(0, "");
}

} // namespace bunruija
