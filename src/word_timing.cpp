#include "word_timing.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace crisperwhisper::detail {
namespace {

constexpr int kSpaceToken = 220;
constexpr double kFrameDurationSeconds = 0.02;
constexpr float kNegativeInfinity = -1.0e9F;

std::string overlap_key(const std::string & word) {
    std::string key;
    key.reserve(word.size());
    for (const unsigned char character : word) {
        if (std::isalnum(character) != 0) {
            key.push_back(
                static_cast<char>(std::tolower(character))
            );
        }
    }
    return key;
}

bool is_special_piece(const std::string & piece) {
    if (piece.size() >= 4 && piece.rfind("<|", 0) == 0 &&
        piece.compare(piece.size() - 2, 2, "|>") == 0) {
        return true;
    }
    if (piece.rfind("[verbatim_", 0) == 0 ||
        piece.rfind("[intended_", 0) == 0) {
        return true;
    }
    static const std::unordered_set<std::string> markers = {
        "<ctx>", "<ectx>", "<htx>", "<ehtx>",
        "<vtx>", "<evtx>", "<sot>", "<eot>",
    };
    return markers.count(piece) != 0;
}

bool is_space_piece(const int token_id, const std::string & piece) {
    return token_id == kSpaceToken || piece == " ";
}

struct WordGroups {
    std::vector<std::vector<std::size_t>> token_indices;
    std::vector<std::string> text;
};

std::string trim_copy(const std::string & text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

WordGroups group_tokens_into_words(
    const std::vector<int> & token_ids,
    const std::vector<std::string> & pieces
) {
    WordGroups result;
    std::vector<std::size_t> current_indices;
    std::string current_text;

    const auto flush = [&]() {
        const std::string trimmed = trim_copy(current_text);
        if (!trimmed.empty() && !current_indices.empty()) {
            result.token_indices.push_back(current_indices);
            result.text.push_back(trimmed);
        }
        current_indices.clear();
        current_text.clear();
    };

    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        const std::string & piece = pieces[i];
        if (is_special_piece(piece)) {
            flush();
            continue;
        }
        if (is_space_piece(token_ids[i], piece)) {
            flush();
            continue;
        }
        if (!current_text.empty() && !piece.empty() && piece.front() == ' ') {
            flush();
        }
        current_indices.push_back(i);
        current_text += piece;
    }
    flush();
    return result;
}

float percentile(std::vector<float> values, const double fraction) {
    if (values.empty()) {
        return 0.0F;
    }
    std::sort(values.begin(), values.end());
    const double position =
        fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return static_cast<float>(
        static_cast<double>(values[lower]) * (1.0 - weight) +
        static_cast<double>(values[upper]) * weight
    );
}

std::vector<float> resample_1d(
    const std::vector<float> & source,
    const int target_length
) {
    if (target_length <= 0) {
        return {};
    }
    if (source.empty()) {
        return std::vector<float>(
            static_cast<std::size_t>(target_length), 0.0F
        );
    }
    if (source.size() == 1) {
        return std::vector<float>(
            static_cast<std::size_t>(target_length), source.front()
        );
    }
    if (source.size() == static_cast<std::size_t>(target_length)) {
        return source;
    }

    std::vector<float> output(static_cast<std::size_t>(target_length));
    if (target_length == 1) {
        output.front() = source.front();
        return output;
    }
    for (int i = 0; i < target_length; ++i) {
        const double position =
            static_cast<double>(i) *
            static_cast<double>(source.size() - 1) /
            static_cast<double>(target_length - 1);
        const auto lower = static_cast<std::size_t>(std::floor(position));
        const auto upper = std::min(lower + 1, source.size() - 1);
        const double weight = position - static_cast<double>(lower);
        output[static_cast<std::size_t>(i)] = static_cast<float>(
            static_cast<double>(source[lower]) * (1.0 - weight) +
            static_cast<double>(source[upper]) * weight
        );
    }
    return output;
}

std::vector<float> token_log_probabilities(
    const std::vector<float> & attention,
    const std::size_t token_count,
    const int frame_count
) {
    constexpr double sharpen = 5.0;
    constexpr double epsilon = 1.0e-8;
    std::vector<float> output(attention.size());

    for (std::size_t token = 0; token < token_count; ++token) {
        const std::size_t offset =
            token * static_cast<std::size_t>(frame_count);
        double sum = 0.0;
        for (int frame = 0; frame < frame_count; ++frame) {
            const double value = std::pow(
                std::max(
                    0.0F,
                    attention[offset + static_cast<std::size_t>(frame)]
                ),
                sharpen
            );
            output[offset + static_cast<std::size_t>(frame)] =
                static_cast<float>(value);
            sum += value;
        }
        sum = std::max(sum, epsilon);
        for (int frame = 0; frame < frame_count; ++frame) {
            const std::size_t index =
                offset + static_cast<std::size_t>(frame);
            output[index] = static_cast<float>(
                std::log(
                    static_cast<double>(output[index]) / sum + epsilon
                )
            );
        }
    }
    return output;
}

std::vector<float> blank_log_probabilities(
    const std::vector<float> & mel,
    const int mel_bins,
    const int mel_frames,
    const int target_frames
) {
    std::vector<float> energy(static_cast<std::size_t>(mel_frames), 0.0F);
    for (int frame = 0; frame < mel_frames; ++frame) {
        double total = 0.0;
        for (int bin = 0; bin < mel_bins; ++bin) {
            total += mel[
                static_cast<std::size_t>(bin) *
                    static_cast<std::size_t>(mel_frames) +
                static_cast<std::size_t>(frame)
            ];
        }
        energy[static_cast<std::size_t>(frame)] =
            static_cast<float>(total / std::max(mel_bins, 1));
    }

    const float p10 = percentile(energy, 0.10);
    const float p90 = percentile(energy, 0.90);
    const float denominator = std::max(1.0e-6F, p90 - p10);
    for (float & value : energy) {
        value = std::clamp((value - p10) / denominator, 0.0F, 1.0F);
    }

    auto normalized = resample_1d(energy, target_frames);
    for (float & value : normalized) {
        float blank = std::clamp(1.0F - value, 1.0e-4F, 0.9999F);
        blank = std::clamp(
            static_cast<float>(std::pow(blank, 3.0F)),
            1.0e-4F,
            1.0F
        );
        value = std::log(blank + 1.0e-6F) - 3.0F;
    }
    return normalized;
}

float log_add_exp(const float left, const float right) {
    if (left <= kNegativeInfinity) {
        return right;
    }
    if (right <= kNegativeInfinity) {
        return left;
    }
    const float maximum = std::max(left, right);
    return maximum +
        std::log1p(std::exp(-std::abs(left - right)));
}

std::vector<std::pair<std::optional<double>, std::optional<double>>>
align_words_with_blanks(
    const std::vector<float> & token_logp,
    const int frame_count,
    const std::vector<std::vector<std::size_t>> & word_token_indices,
    const std::vector<float> & blank_logp
) {
    const std::size_t word_count = word_token_indices.size();
    if (word_count == 0 || frame_count == 0) {
        return {};
    }

    std::vector<float> word_logp(
        word_count * static_cast<std::size_t>(frame_count),
        kNegativeInfinity
    );
    const std::size_t token_count =
        token_logp.size() / static_cast<std::size_t>(frame_count);
    for (std::size_t word = 0; word < word_count; ++word) {
        for (const std::size_t token : word_token_indices[word]) {
            if (token >= token_count) {
                continue;
            }
            for (int frame = 0; frame < frame_count; ++frame) {
                const std::size_t word_index =
                    word * static_cast<std::size_t>(frame_count) +
                    static_cast<std::size_t>(frame);
                const std::size_t token_index =
                    token * static_cast<std::size_t>(frame_count) +
                    static_cast<std::size_t>(frame);
                word_logp[word_index] = log_add_exp(
                    word_logp[word_index], token_logp[token_index]
                );
            }
        }
    }

    const std::size_t state_count = 2 * word_count + 1;
    std::vector<float> previous(state_count, kNegativeInfinity);
    std::vector<float> current(state_count, kNegativeInfinity);
    std::vector<std::uint8_t> back(
        state_count * static_cast<std::size_t>(frame_count), 0
    );
    previous[0] = blank_logp[0];

    for (int frame = 1; frame < frame_count; ++frame) {
        for (std::size_t state = 0; state < state_count; ++state) {
            const float stay = previous[state];
            const float advance =
                state == 0 ? kNegativeInfinity : previous[state - 1];
            const bool take_advance = advance > stay;
            const float emission = state % 2 == 0
                ? blank_logp[static_cast<std::size_t>(frame)]
                : word_logp[
                    (state / 2) * static_cast<std::size_t>(frame_count) +
                    static_cast<std::size_t>(frame)
                ];
            current[state] =
                (take_advance ? advance : stay) + emission;
            back[
                static_cast<std::size_t>(frame) * state_count + state
            ] = take_advance ? 1 : 0;
        }
        previous.swap(current);
        std::fill(current.begin(), current.end(), kNegativeInfinity);
    }

    std::size_t state = 0;
    float best = kNegativeInfinity;
    for (std::size_t candidate = 0;
         candidate < state_count;
         ++candidate) {
        const float score =
            previous[candidate] +
            static_cast<float>(candidate) * 1.0e-4F;
        if (score > best) {
            best = score;
            state = candidate;
        }
    }

    std::vector<std::size_t> states(
        static_cast<std::size_t>(frame_count), 0
    );
    for (int frame = frame_count - 1; frame >= 0; --frame) {
        states[static_cast<std::size_t>(frame)] = state;
        if (frame > 0 &&
            back[
                static_cast<std::size_t>(frame) * state_count + state
            ] != 0) {
            --state;
        }
    }

    std::vector<std::pair<std::optional<double>, std::optional<double>>>
        timings(word_count);
    for (std::size_t word = 0; word < word_count; ++word) {
        const std::size_t word_state = 2 * word + 1;
        int first = -1;
        int last = -1;
        for (int frame = 0; frame < frame_count; ++frame) {
            if (states[static_cast<std::size_t>(frame)] == word_state) {
                if (first < 0) {
                    first = frame;
                }
                last = frame;
            }
        }
        if (first >= 0) {
            timings[word] = {
                static_cast<double>(first) * kFrameDurationSeconds,
                static_cast<double>(last) * kFrameDurationSeconds,
            };
        }
    }
    return timings;
}

void split_short_gaps(std::vector<AlignedWord> & words) {
    constexpr double maximum_gap = 0.1;
    std::optional<std::size_t> previous;
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (!words[i].start_seconds.has_value() ||
            !words[i].end_seconds.has_value()) {
            continue;
        }
        if (previous.has_value()) {
            auto & left = words[*previous];
            auto & right = words[i];
            const double gap =
                *right.start_seconds - *left.end_seconds;
            if (gap > 0.0 && gap <= maximum_gap) {
                const double middle = *left.end_seconds + gap / 2.0;
                left.end_seconds = middle;
                right.start_seconds = middle;
            }
        }
        previous = i;
    }
}

} // namespace

std::size_t duplicate_prefix_words(
    const std::vector<std::string> & confirmed,
    const std::vector<std::string> & current,
    const std::size_t current_end
) {
    const std::size_t usable_end =
        std::min(current_end, current.size());
    if (confirmed.empty() || usable_end == 0) {
        return 0;
    }

    const std::size_t search_limit = std::min<std::size_t>(
        32, std::min(confirmed.size(), usable_end)
    );
    const std::size_t maximum_offset =
        std::min<std::size_t>(8, usable_end - 1);
    std::size_t best_length = 0;
    std::size_t best_consumed = 0;

    for (std::size_t offset = 0;
         offset <= maximum_offset;
        ++offset) {
        const std::size_t maximum_length = std::min(
            search_limit, usable_end - offset
        );
        for (std::size_t length = 1;
             length <= maximum_length;
             ++length) {
            bool equal = true;
            for (std::size_t i = 0; i < length; ++i) {
                const auto & left =
                    confirmed[confirmed.size() - length + i];
                const auto & right = current[offset + i];
                if (overlap_key(left) != overlap_key(right)) {
                    equal = false;
                    break;
                }
            }
            if (!equal) {
                continue;
            }

            const std::string single_key = overlap_key(
                current[offset + length - 1]
            );
            if (length == 1 && single_key.size() < 5) {
                continue;
            }
            const std::size_t consumed = offset + length;
            if (length > best_length ||
                (length == best_length && consumed > best_consumed)) {
                best_length = length;
                best_consumed = consumed;
            }
        }
    }
    return best_consumed;
}

bool equivalent_word(
    const std::string & left,
    const std::string & right
) {
    const std::string left_key = overlap_key(left);
    return !left_key.empty() && left_key == overlap_key(right);
}

std::vector<AlignedWord> extract_word_timings(
    const std::vector<int> & token_ids,
    const std::vector<std::string> & token_pieces,
    const std::vector<float> & attention,
    const int attention_frames,
    const std::vector<float> & mel,
    const int mel_bins,
    const int mel_frames
) {
    if (token_ids.size() != token_pieces.size()) {
        throw std::invalid_argument(
            "token ids and token pieces must be one-to-one"
        );
    }
    if (attention_frames <= 0 || mel_bins <= 0 || mel_frames <= 0) {
        return {};
    }
    if (attention.size() !=
        token_ids.size() * static_cast<std::size_t>(attention_frames)) {
        throw std::invalid_argument(
            "attention shape does not match generated tokens"
        );
    }
    if (mel.size() !=
        static_cast<std::size_t>(mel_bins) *
        static_cast<std::size_t>(mel_frames)) {
        throw std::invalid_argument("mel shape is inconsistent");
    }
    if (token_ids.empty()) {
        return {};
    }

    const WordGroups groups =
        group_tokens_into_words(token_ids, token_pieces);
    if (groups.token_indices.empty()) {
        return {};
    }

    const auto token_logp = token_log_probabilities(
        attention, token_ids.size(), attention_frames
    );
    const auto blank_logp = blank_log_probabilities(
        mel, mel_bins, mel_frames, attention_frames
    );
    const auto timings = align_words_with_blanks(
        token_logp,
        attention_frames,
        groups.token_indices,
        blank_logp
    );

    std::vector<AlignedWord> result;
    result.reserve(groups.text.size());
    for (std::size_t i = 0; i < groups.text.size(); ++i) {
        result.push_back({
            groups.text[i],
            timings[i].first,
            timings[i].second,
        });
    }
    split_short_gaps(result);
    return result;
}

void monotonize_words(std::vector<AlignedWord> & words) {
    std::optional<double> previous_end;
    for (auto & word : words) {
        if (!word.start_seconds.has_value() ||
            !word.end_seconds.has_value()) {
            continue;
        }
        if (previous_end.has_value() &&
            *word.start_seconds < *previous_end) {
            word.start_seconds = *previous_end;
            word.end_seconds = std::max(
                *previous_end, *word.end_seconds
            );
        }
        previous_end = word.end_seconds;
    }
}

} // namespace crisperwhisper::detail
