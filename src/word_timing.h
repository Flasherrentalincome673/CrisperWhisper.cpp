#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace crisperwhisper::detail {

struct AlignedWord {
    std::string word;
    std::optional<double> start_seconds;
    std::optional<double> end_seconds;
};

// Port of CrisperWhisper's supervised cross-attention word aligner.
//
// attention is row-major [token_count, attention_frames].
// mel is row-major [mel_bins, mel_frames].
std::vector<AlignedWord> extract_word_timings(
    const std::vector<int> & token_ids,
    const std::vector<std::string> & token_pieces,
    const std::vector<float> & attention,
    int attention_frames,
    const std::vector<float> & mel,
    int mel_bins,
    int mel_frames
);

void monotonize_words(std::vector<AlignedWord> & words);

// Returns the number of leading current words already represented by the
// confirmed suffix. Used as the text-only seam fallback for long audio.
std::size_t duplicate_prefix_words(
    const std::vector<std::string> & confirmed,
    const std::vector<std::string> & current,
    std::size_t current_end
);

bool equivalent_word(
    const std::string & left,
    const std::string & right
);

} // namespace crisperwhisper::detail
