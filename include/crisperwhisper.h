#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace crisperwhisper {

enum class Mode {
    Verbatim,
    Intended,
    Verbatimize,
};

struct TranscriptionOptions {
    Mode mode = Mode::Verbatim;
    std::string language = "en";
    std::vector<std::string> hotwords;

    int threads = 4;
    int max_new_tokens = 256;

    // CrisperWhisper's continuation long-form defaults.
    double chunk_seconds = 30.0;
    double stride_seconds = 26.0;
    int context_words = 12;
    int drop_words = 2;

    // Opt-in supervised word alignment from CrisperWhisper's trained
    // decoder-to-audio attention heads. This loads a separate no-flash
    // alignment context on first use; normal transcription keeps flash
    // attention enabled.
    bool word_timestamps = false;

    // When set, switches from transcription to the "verbatimize" task.
    std::optional<std::string> verbatimize_transcript;
};

struct WordTimestamp {
    std::string word;
    double start_seconds = 0.0;
    double end_seconds = 0.0;
};

struct ChunkResult {
    int index = 0;
    double start_seconds = 0.0;
    double end_seconds = 0.0;
    std::string text;
    std::string context;
    bool is_last = false;
    std::vector<WordTimestamp> words;
};

struct TranscriptionResult {
    std::string text;
    std::string language;
    Mode mode = Mode::Verbatim;
    bool word_timestamps = false;
    double duration_seconds = 0.0;
    double processing_seconds = 0.0;
    std::vector<ChunkResult> chunks;
    std::vector<WordTimestamp> words;
};

struct ModelOptions {
    bool use_gpu = true;
    bool flash_attention = true;
    int gpu_device = 0;
};

// Reusable native C++ model.  The model file is the output of
// tools/convert_hf_to_ggml.py.
class Model {
public:
    explicit Model(const std::string & model_path, const ModelOptions & options = {});
    ~Model();

    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;

    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    TranscriptionResult transcribe_file(
        const std::string & audio_path,
        const TranscriptionOptions & options = {}
    );

    TranscriptionResult transcribe(
        const std::vector<float> & mono_16khz_audio,
        const TranscriptionOptions & options = {}
    );

    int vocabulary_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char * mode_name(Mode mode);

// Exposed for callers that want the same whitespace cleanup as the CLI.
std::string normalize_transcript_whitespace(const std::string & text);

} // namespace crisperwhisper
