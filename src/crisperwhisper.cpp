#include "crisperwhisper.h"

#include "audio.h"
#include "whisper.h"
#include "word_timing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace crisperwhisper {
namespace {

constexpr int kSampleRate = 16'000;

// generation_config.json from the open CrisperWhisper 2.0 checkpoints.  These
// are ordinary text/audio tokens that Whisper suppresses during generation,
// plus control tokens.  Negative sentinels are intentionally absent.
const std::unordered_set<int> kDefaultSuppressTokens = {
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
    63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922,
    931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846,
    3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929,
    11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618,
    16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161,
    26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425,
    49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362,
};

using ContextPtr = std::unique_ptr<whisper_context, decltype(&whisper_free)>;
using StatePtr = std::unique_ptr<whisper_state, decltype(&whisper_free_state)>;

std::vector<std::string> split_words(const std::string & text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::string join_words(
    const std::vector<std::string> & words,
    std::size_t begin,
    std::size_t end
) {
    std::ostringstream stream;
    for (std::size_t i = begin; i < end; ++i) {
        if (i != begin) {
            stream << ' ';
        }
        stream << words[i];
    }
    return stream.str();
}

bool is_control_token_text(const std::string & token) {
    if (token.size() >= 4 && token.rfind("<|", 0) == 0 &&
        token.compare(token.size() - 2, 2, "|>") == 0) {
        return true;
    }
    if (token.rfind("[verbatim_", 0) == 0 ||
        token.rfind("[intended_", 0) == 0) {
        return true;
    }
    static const std::unordered_set<std::string> markers = {
        "<vtx>", "<evtx>", "<ctx>", "<ectx>", "<htx>", "<ehtx>",
    };
    return markers.count(token) != 0;
}

} // namespace

struct Model::Impl {
    std::string model_path;
    ModelOptions model_options;
    ContextPtr context{nullptr, &whisper_free};
    mutable ContextPtr timing_context{nullptr, &whisper_free};
    mutable std::vector<whisper_ahead> timing_heads;
    int n_vocab = 0;

    int eot = -1;
    int sot = -1;
    int transcribe = -1;
    int no_timestamps = -1;
    std::vector<int> verbatim_tags;
    std::vector<int> intended_tags;
    int vtx = -1;
    int evtx = -1;
    int ctx_start = -1;
    int ctx_end = -1;
    int hotword_start = -1;
    int hotword_end = -1;

    explicit Impl(
        const std::string & model_path_value,
        const ModelOptions & options
    )
        : model_path(model_path_value),
          model_options(options) {
        whisper_context_params params = whisper_context_default_params();
        params.use_gpu = options.use_gpu;
        params.flash_attn = options.flash_attention;
        params.gpu_device = options.gpu_device;

        context.reset(whisper_init_from_file_with_params_no_state(
            model_path.c_str(), params
        ));
        if (!context) {
            throw std::runtime_error(
                "could not load model '" + model_path +
                "'; use tools/convert_hf_to_ggml.py to create it"
            );
        }

        n_vocab = whisper_model_n_vocab(context.get());
        std::unordered_map<std::string, int> ids;
        ids.reserve(static_cast<std::size_t>(n_vocab));
        for (int id = 0; id < n_vocab; ++id) {
            const char * raw = whisper_token_to_str(context.get(), id);
            if (raw != nullptr) {
                ids[raw] = id;
            }
        }

        const auto require = [&ids](const std::string & token) {
            const auto it = ids.find(token);
            if (it == ids.end()) {
                throw std::runtime_error(
                    "model vocabulary is missing required CrisperWhisper "
                    "token: " + token +
                    ". Reconvert it with this repository's converter."
                );
            }
            return it->second;
        };

        eot = require("<|endoftext|>");
        sot = require("<|startoftranscript|>");
        transcribe = require("<|transcribe|>");
        no_timestamps = require("<|notimestamps|>");

        for (int i = 1; i <= 5; ++i) {
            verbatim_tags.push_back(require(
                "[verbatim_" + std::to_string(i) + "]"
            ));
            intended_tags.push_back(require(
                "[intended_" + std::to_string(i) + "]"
            ));
        }

        vtx = require("<vtx>");
        evtx = require("<evtx>");
        ctx_start = require("<ctx>");
        ctx_end = require("<ectx>");
        hotword_start = require("<htx>");
        hotword_end = require("<ehtx>");
    }

    int find_token(const std::string & token) const {
        for (int id = 0; id < n_vocab; ++id) {
            const char * raw = whisper_token_to_str(context.get(), id);
            if (raw != nullptr && token == raw) {
                return id;
            }
        }
        return -1;
    }

    std::vector<int> tokenize_regular_text(const std::string & text) const {
        if (text.empty()) {
            return {};
        }
        if (text.size() >=
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("decoder prompt text is too large");
        }

        // A byte-level BPE cannot emit more tokens than input bytes. Starting
        // with that capacity avoids whisper_tokenize's intentionally noisy
        // size-probe path during normal CLI use.
        int capacity = static_cast<int>(text.size()) + 1;
        std::vector<int> tokens(static_cast<std::size_t>(capacity));
        int actual = whisper_tokenize(
            context.get(), text.c_str(), tokens.data(), capacity
        );
        if (actual < 0) {
            capacity = -actual;
            tokens.resize(static_cast<std::size_t>(capacity));
            actual = whisper_tokenize(
                context.get(), text.c_str(), tokens.data(), capacity
            );
            if (actual < 0) {
                throw std::runtime_error(
                    "token buffer was unexpectedly too small"
                );
            }
        }
        tokens.resize(static_cast<std::size_t>(actual));
        return tokens;
    }

    static void append(std::vector<int> & destination, const std::vector<int> & source) {
        destination.insert(destination.end(), source.begin(), source.end());
    }

    std::vector<int> make_prompt(
        const TranscriptionOptions & options,
        const std::string & continuation_context
    ) const {
        std::vector<int> prompt =
            options.mode == Mode::Intended ? intended_tags : verbatim_tags;

        if (options.verbatimize_transcript.has_value()) {
            // Matches:
            // [verbatim_1]...[verbatim_5] <vtx> transcript <evtx>
            append(prompt, tokenize_regular_text(" "));
            prompt.push_back(vtx);
            append(prompt, tokenize_regular_text(
                " " + *options.verbatimize_transcript + " "
            ));
            prompt.push_back(evtx);
        } else {
            if (!continuation_context.empty()) {
                append(prompt, tokenize_regular_text(" "));
                prompt.push_back(ctx_start);
                append(prompt, tokenize_regular_text(
                    " " + continuation_context + " "
                ));
                prompt.push_back(ctx_end);
            }

            if (!options.hotwords.empty()) {
                append(prompt, tokenize_regular_text(" "));
                prompt.push_back(hotword_start);
                std::ostringstream text;
                for (std::size_t i = 0; i < options.hotwords.size(); ++i) {
                    if (i != 0) {
                        text << ' ';
                    }
                    text << options.hotwords[i];
                }
                append(prompt, tokenize_regular_text(" " + text.str() + " "));
                prompt.push_back(hotword_end);
            }
        }

        const int language = find_token("<|" + options.language + "|>");
        if (language < 0) {
            throw std::invalid_argument(
                "unsupported language code: " + options.language
            );
        }

        // CrisperWhisper was trained with mode/task material before the
        // standard Whisper decoder prefix.
        prompt.push_back(sot);
        prompt.push_back(language);
        prompt.push_back(transcribe);
        prompt.push_back(no_timestamps);
        return prompt;
    }

    int greedy_token(const float * logits, int generated_count) const {
        int best_id = -1;
        float best_logit = -std::numeric_limits<float>::infinity();

        for (int id = 0; id < n_vocab; ++id) {
            if (kDefaultSuppressTokens.count(id) != 0) {
                continue;
            }
            // Match the Python engines: EOT is legal on the first step.
            (void)generated_count;
            if (logits[id] > best_logit) {
                best_logit = logits[id];
                best_id = id;
            }
        }
        if (best_id < 0) {
            throw std::runtime_error("decoder did not produce a usable token");
        }
        return best_id;
    }

    std::vector<whisper_ahead> crisper_alignment_heads() const {
        const int layers = whisper_model_n_text_layer(context.get());
        const int heads = whisper_model_n_text_head(context.get());
        const int mels = whisper_model_n_mels(context.get());

        // generation_config.json from each open CrisperWhisper 2.0
        // checkpoint. These are supervised model-specific heads, not
        // whisper.cpp's generic DTW presets.
        if (layers == 12 && heads == 12 && mels == 80) {
            return {
                {7, 8}, {3, 10}, {4, 0}, {7, 4}, {5, 3},
                {0, 11}, {8, 8}, {7, 9}, {4, 10}, {5, 2},
            };
        }
        if (layers == 24 && heads == 16 && mels == 80) {
            return {
                {7, 15}, {5, 0}, {1, 6}, {7, 5}, {8, 10},
                {9, 0}, {6, 2}, {8, 6}, {10, 12}, {17, 0},
            };
        }
        if (layers == 4 && heads == 20 && mels == 128) {
            return {
                {2, 11}, {1, 1}, {2, 8}, {3, 12}, {3, 15},
                {3, 6}, {3, 17}, {2, 4}, {3, 3}, {3, 18},
            };
        }
        if (layers == 32 && heads == 20 && mels == 80) {
            return {
                {8, 8}, {10, 3}, {4, 11}, {9, 18}, {11, 16},
                {18, 14}, {5, 16}, {15, 4}, {20, 16}, {16, 11},
            };
        }

        throw std::runtime_error(
            "word timestamps are not configured for this model shape "
            "(layers=" + std::to_string(layers) +
            ", heads=" + std::to_string(heads) +
            ", mels=" + std::to_string(mels) + ")"
        );
    }

    whisper_context * ensure_timing_context() const {
        if (timing_context) {
            return timing_context.get();
        }

        timing_heads = crisper_alignment_heads();
        whisper_context_params params = whisper_context_default_params();
        params.use_gpu = model_options.use_gpu;
        params.flash_attn = false;
        params.gpu_device = model_options.gpu_device;
        params.dtw_token_timestamps = true;
        params.dtw_aheads_preset = WHISPER_AHEADS_CUSTOM;
        params.dtw_aheads.n_heads = timing_heads.size();
        params.dtw_aheads.heads = timing_heads.data();

        timing_context.reset(whisper_init_from_file_with_params_no_state(
            model_path.c_str(), params
        ));
        if (!timing_context) {
            throw std::runtime_error(
                "could not load the word-timestamp alignment context"
            );
        }
        return timing_context.get();
    }

    std::vector<int> decode_chunk(
        const float * samples,
        int sample_count,
        const std::vector<int> & prompt,
        int threads,
        int max_new_tokens
    ) const {
        StatePtr state(whisper_init_state(context.get()), &whisper_free_state);
        if (!state) {
            throw std::runtime_error("could not allocate whisper decoder state");
        }

        if (whisper_pcm_to_mel_with_state(
                context.get(), state.get(), samples, sample_count, threads
            ) != 0) {
            throw std::runtime_error("failed to compute log-mel spectrogram");
        }
        if (whisper_encode_with_state(
                context.get(), state.get(), 0, threads
            ) != 0) {
            throw std::runtime_error("failed to run audio encoder");
        }

        const int text_context = whisper_model_n_text_ctx(context.get());
        if (prompt.empty() ||
            static_cast<int>(prompt.size()) >= text_context) {
            throw std::runtime_error("decoder prompt exceeds model context");
        }

        if (whisper_decode_with_state(
                context.get(),
                state.get(),
                prompt.data(),
                static_cast<int>(prompt.size()),
                0,
                threads
            ) != 0) {
            throw std::runtime_error("failed to evaluate decoder prompt");
        }

        std::vector<int> generated;
        generated.reserve(static_cast<std::size_t>(max_new_tokens));
        int n_past = static_cast<int>(prompt.size());

        for (int step = 0;
             step < max_new_tokens && n_past < text_context;
             ++step) {
            float * all_logits = whisper_get_logits_from_state(state.get());
            if (all_logits == nullptr) {
                throw std::runtime_error("decoder returned no logits");
            }

            const int row =
                step == 0 ? static_cast<int>(prompt.size()) - 1 : 0;
            const float * logits =
                all_logits + static_cast<std::ptrdiff_t>(row) * n_vocab;
            const int token = greedy_token(logits, step);
            if (token == eot) {
                break;
            }

            generated.push_back(token);
            if (step + 1 >= max_new_tokens || n_past + 1 >= text_context) {
                break;
            }

            if (whisper_decode_with_state(
                    context.get(),
                    state.get(),
                    &token,
                    1,
                    n_past,
                    threads
                ) != 0) {
                throw std::runtime_error("failed during autoregressive decode");
            }
            ++n_past;
        }

        return generated;
    }

    std::vector<detail::AlignedWord> align_chunk_words(
        const float * samples,
        int sample_count,
        const std::vector<int> & prompt,
        const std::vector<int> & generated,
        int threads
    ) const {
        if (generated.empty()) {
            return {};
        }

        whisper_context * timing = ensure_timing_context();
        StatePtr state(
            whisper_init_state(timing), &whisper_free_state
        );
        if (!state) {
            throw std::runtime_error(
                "could not allocate word-timestamp decoder state"
            );
        }
        if (whisper_pcm_to_mel_with_state(
                timing, state.get(), samples, sample_count, threads
            ) != 0) {
            throw std::runtime_error(
                "failed to compute the word-timestamp spectrogram"
            );
        }
        if (whisper_encode_with_state(
                timing, state.get(), 0, threads
            ) != 0) {
            throw std::runtime_error(
                "failed to run the word-timestamp audio encoder"
            );
        }

        // Decoder row prompt.size()-1 predicts generated[0]. Feeding all
        // known generated tokens except the last captures one row per output
        // token in a single teacher-forced pass.
        std::vector<int> teacher_tokens = prompt;
        teacher_tokens.insert(
            teacher_tokens.end(), generated.begin(), generated.end() - 1
        );
        if (teacher_tokens.size() >
            static_cast<std::size_t>(
                whisper_model_n_text_ctx(timing)
            )) {
            throw std::runtime_error(
                "word-timestamp sequence exceeds model context"
            );
        }
        if (whisper_decode_with_state_and_cross_attention(
                timing,
                state.get(),
                teacher_tokens.data(),
                static_cast<int>(teacher_tokens.size()),
                0,
                threads
            ) != 0) {
            throw std::runtime_error(
                "failed to capture supervised cross-attention"
            );
        }

        const int captured_tokens =
            whisper_cross_attention_n_tokens_from_state(state.get());
        const int attention_frames =
            whisper_cross_attention_n_frames_from_state(state.get());
        const int selected_heads =
            whisper_cross_attention_n_heads_from_state(state.get());
        const std::size_t first_generated_row = prompt.size() - 1;
        if (captured_tokens <= 0 || attention_frames <= 0 ||
            selected_heads <= 0 ||
            first_generated_row + generated.size() >
                static_cast<std::size_t>(captured_tokens)) {
            throw std::runtime_error(
                "captured cross-attention has an unexpected shape"
            );
        }

        const std::size_t raw_count =
            static_cast<std::size_t>(captured_tokens) *
            static_cast<std::size_t>(attention_frames) *
            static_cast<std::size_t>(selected_heads);
        std::vector<float> raw_attention(raw_count);
        if (whisper_cross_attention_copy_from_state(
                state.get(), raw_attention.data(), raw_attention.size()
            ) != raw_attention.size()) {
            throw std::runtime_error(
                "failed to copy supervised cross-attention"
            );
        }

        std::vector<float> attention(
            generated.size() *
            static_cast<std::size_t>(attention_frames),
            0.0F
        );
        for (std::size_t token = 0; token < generated.size(); ++token) {
            const std::size_t source_token =
                first_generated_row + token;
            for (int frame = 0; frame < attention_frames; ++frame) {
                double total = 0.0;
                for (int head = 0; head < selected_heads; ++head) {
                    const std::size_t source =
                        source_token +
                        static_cast<std::size_t>(captured_tokens) *
                            static_cast<std::size_t>(frame) +
                        static_cast<std::size_t>(captured_tokens) *
                            static_cast<std::size_t>(attention_frames) *
                            static_cast<std::size_t>(head);
                    total += raw_attention[source];
                }
                attention[
                    token * static_cast<std::size_t>(attention_frames) +
                    static_cast<std::size_t>(frame)
                ] = static_cast<float>(
                    total / static_cast<double>(selected_heads)
                );
            }
        }

        const int mel_frames =
            whisper_mel_n_frames_from_state(state.get());
        const int mel_bins =
            whisper_mel_n_mels_from_state(state.get());
        const std::size_t mel_count =
            static_cast<std::size_t>(mel_frames) *
            static_cast<std::size_t>(mel_bins);
        std::vector<float> mel(mel_count);
        if (mel_count == 0 ||
            whisper_mel_copy_from_state(
                state.get(), mel.data(), mel.size()
            ) != mel.size()) {
            throw std::runtime_error(
                "failed to copy the word-timestamp spectrogram"
            );
        }

        std::vector<std::string> pieces;
        pieces.reserve(generated.size());
        for (const int token : generated) {
            const char * raw =
                whisper_token_to_str(context.get(), token);
            pieces.emplace_back(raw == nullptr ? "" : raw);
        }
        return detail::extract_word_timings(
            generated,
            pieces,
            attention,
            attention_frames,
            mel,
            mel_bins,
            mel_frames
        );
    }

    std::string decode_text(const std::vector<int> & tokens) const {
        std::string text;
        for (const int token : tokens) {
            if (token == eot) {
                break;
            }
            const char * raw = whisper_token_to_str(context.get(), token);
            if (raw == nullptr) {
                continue;
            }
            const std::string piece(raw);
            if (!is_control_token_text(piece)) {
                text += piece;
            }
        }
        return normalize_transcript_whitespace(text);
    }
};

Model::Model(const std::string & model_path, const ModelOptions & options)
    : impl_(std::make_unique<Impl>(model_path, options)) {}

Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;

int Model::vocabulary_size() const {
    return impl_->n_vocab;
}

TranscriptionResult Model::transcribe_file(
    const std::string & audio_path,
    const TranscriptionOptions & options
) {
    return transcribe(detail::load_audio_16khz_mono(audio_path), options);
}

TranscriptionResult Model::transcribe(
    const std::vector<float> & audio,
    const TranscriptionOptions & options
) {
    if (audio.empty()) {
        throw std::invalid_argument("audio contains no samples");
    }
    if (options.threads < 1) {
        throw std::invalid_argument("threads must be at least 1");
    }
    if (options.max_new_tokens < 1) {
        throw std::invalid_argument("max_new_tokens must be at least 1");
    }
    if (options.mode == Mode::Verbatimize &&
        !options.verbatimize_transcript.has_value()) {
        throw std::invalid_argument(
            "verbatimize mode requires an intended transcript"
        );
    }
    if (options.chunk_seconds <= 0.0 || options.chunk_seconds > 30.0) {
        throw std::invalid_argument("chunk_seconds must be in (0, 30]");
    }
    if (options.stride_seconds <= 0.0 ||
        options.stride_seconds > options.chunk_seconds) {
        throw std::invalid_argument(
            "stride_seconds must be in (0, chunk_seconds]"
        );
    }
    if (options.context_words < 0 || options.drop_words < 0) {
        throw std::invalid_argument(
            "context_words and drop_words cannot be negative"
        );
    }

    const auto started = std::chrono::steady_clock::now();
    TranscriptionResult result;
    result.language = options.language;
    result.mode = options.verbatimize_transcript.has_value()
        ? Mode::Verbatimize
        : options.mode;
    result.word_timestamps = options.word_timestamps;
    result.duration_seconds =
        static_cast<double>(audio.size()) / kSampleRate;

    const auto chunk_samples = static_cast<std::size_t>(
        std::llround(options.chunk_seconds * kSampleRate)
    );
    const auto stride_samples = static_cast<std::size_t>(
        std::llround(options.stride_seconds * kSampleRate)
    );

    std::vector<std::string> confirmed_words;
    std::size_t start = 0;
    int chunk_index = 0;

    while (start < audio.size()) {
        const std::size_t end =
            std::min(start + chunk_samples, audio.size());
        const bool is_last = end == audio.size();

        std::string continuation_context;
        if (chunk_index > 0 && !confirmed_words.empty()) {
            const auto wanted =
                static_cast<std::size_t>(options.context_words);
            const std::size_t context_begin =
                confirmed_words.size() > wanted
                    ? confirmed_words.size() - wanted
                    : 0;
            continuation_context = join_words(
                confirmed_words, context_begin, confirmed_words.size()
            );
        }

        const auto prompt =
            impl_->make_prompt(options, continuation_context);
        const auto tokens = impl_->decode_chunk(
            audio.data() + start,
            static_cast<int>(end - start),
            prompt,
            options.threads,
            options.max_new_tokens
        );
        const std::string raw_text = impl_->decode_text(tokens);
        const auto words = split_words(raw_text);
        std::vector<detail::AlignedWord> aligned_words;
        if (options.word_timestamps) {
            aligned_words = impl_->align_chunk_words(
                audio.data() + start,
                static_cast<int>(end - start),
                prompt,
                tokens,
                options.threads
            );
        }

        std::size_t keep = words.size();
        if (!is_last) {
            const auto drop =
                static_cast<std::size_t>(options.drop_words);
            keep = words.size() > drop ? words.size() - drop : 0;
        }
        if (options.word_timestamps && !is_last) {
            // Give each stride interval to exactly one chunk. Everything
            // beginning in this chunk's trailing overlap is re-covered by
            // the next window, so it must not be emitted twice.
            for (std::size_t i = 0; i < aligned_words.size(); ++i) {
                if (aligned_words[i].start_seconds.has_value() &&
                    *aligned_words[i].start_seconds >=
                        options.stride_seconds) {
                    keep = std::min(words.size(), i);
                    break;
                }
            }
        }

        const double chunk_start_seconds =
            static_cast<double>(start) / kSampleRate;
        std::size_t prefix_skip = 0;
        if (chunk_index > 0 && options.word_timestamps &&
            !result.words.empty()) {
            // Catch the one word that can straddle the ownership cutoff.
            // Requiring both equivalent text and coincident absolute time
            // prevents repeated speech from being deleted as false overlap.
            prefix_skip = detail::timestamp_duplicate_prefix_words(
                aligned_words,
                keep,
                chunk_start_seconds,
                result.words.back().word,
                result.words.back().end_seconds
            );
        } else if (chunk_index > 0) {
            // Timings were not requested, so retain the legacy text-only
            // seam fallback.
            prefix_skip = detail::duplicate_prefix_words(
                confirmed_words, words, keep
            );
        }
        prefix_skip = std::min(prefix_skip, keep);

        confirmed_words.insert(
            confirmed_words.end(),
            words.begin() + prefix_skip,
            words.begin() + keep
        );

        ChunkResult chunk;
        chunk.index = chunk_index;
        chunk.start_seconds = chunk_start_seconds;
        chunk.end_seconds =
            static_cast<double>(end) / kSampleRate;
        chunk.text = join_words(words, prefix_skip, keep);
        chunk.context = continuation_context;
        chunk.is_last = is_last;

        if (options.word_timestamps) {
            const std::size_t aligned_keep =
                std::min(keep, aligned_words.size());
            const double chunk_duration =
                static_cast<double>(end - start) / kSampleRate;
            for (std::size_t i = prefix_skip; i < aligned_keep; ++i) {
                const auto & aligned = aligned_words[i];
                if (!aligned.start_seconds.has_value() ||
                    !aligned.end_seconds.has_value()) {
                    continue;
                }
                double word_start = chunk.start_seconds + std::clamp(
                    *aligned.start_seconds, 0.0, chunk_duration
                );
                double word_end = chunk.start_seconds + std::clamp(
                    *aligned.end_seconds, 0.0, chunk_duration
                );
                if (!result.words.empty() &&
                    word_start < result.words.back().end_seconds) {
                    word_start = result.words.back().end_seconds;
                }
                word_end = std::max(word_start, word_end);

                WordTimestamp timestamp{
                    aligned.word, word_start, word_end,
                };
                chunk.words.push_back(timestamp);
                result.words.push_back(std::move(timestamp));
            }
        }
        result.chunks.push_back(std::move(chunk));

        if (is_last) {
            break;
        }
        start += stride_samples;
        ++chunk_index;
    }

    result.text = join_words(
        confirmed_words, 0, confirmed_words.size()
    );
    result.processing_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    return result;
}

const char * mode_name(const Mode mode) {
    switch (mode) {
    case Mode::Verbatim:
        return "verbatim";
    case Mode::Intended:
        return "intended";
    case Mode::Verbatimize:
        return "verbatimize";
    }
    return "unknown";
}

std::string normalize_transcript_whitespace(const std::string & text) {
    std::istringstream stream(text);
    std::ostringstream normalized;
    std::string word;
    bool first = true;
    while (stream >> word) {
        if (!first) {
            normalized << ' ';
        }
        normalized << word;
        first = false;
    }
    return normalized.str();
}

} // namespace crisperwhisper
