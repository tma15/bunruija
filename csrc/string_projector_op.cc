#include <algorithm>
#include <iostream>

#include "bunruija/string_projector_op.h"


namespace bunruija {


StringProjectorOp::StringProjectorOp(int feature_size, float distortion_probability)
  : hasher_(Hasher(feature_size)),
    feature_size_(feature_size),
    distorter_(StringDistorter(distortion_probability)) {
}


void StringProjectorOp::train(bool mode) {
  is_training_ = mode;
}


bool StringProjectorOp::is_training() {
  return is_training_;
}


void StringProjectorOp::operator()(
    const std::vector<std::vector<std::string>> &batch_words, float *projection) {
  std::vector<uint64_t> hash_codes;

  int max_seq_len = batch_words[0].size();
  for (int batch_index = 1; batch_index < batch_words.size(); ++batch_index) {
    max_seq_len = std::max(max_seq_len, (int)batch_words[batch_index].size());
  }

  for (int batch_index = 0; batch_index < batch_words.size(); ++batch_index) {
    int offset = batch_index * max_seq_len * feature_size_;
    const std::vector<std::string> &words = batch_words[batch_index];

    for (int word_index = 0; word_index < words.size(); ++word_index) {
      const std::string &word = words[word_index];
      hasher_.get_hash_codes(word, &hash_codes);

      for (int hash_index = 0, k = 0; hash_index < hash_codes.size(); ++hash_index) {
        uint64_t hash = hash_codes[hash_index];
        for (int kmax = std::min(k + kincrement, feature_size_); k < kmax;) {
          projection[offset + k] = kmapping_table[hash & 0x3];
          hash >>= 2;
          ++k;
        }
      }
      offset += feature_size_;
    }
  }
}
  
} // namespace bunruija
