#include "crisperwhisper.h"
#include "word_timing.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

int main() {
    using crisperwhisper::normalize_transcript_whitespace;

    assert(normalize_transcript_whitespace("") == "");
    assert(normalize_transcript_whitespace("  hello   world \n") == "hello world");
    assert(
        normalize_transcript_whitespace("[UM]  so,  yes") ==
        "[UM] so, yes"
    );
    assert(std::string(crisperwhisper::mode_name(
        crisperwhisper::Mode::Verbatim
    )) == "verbatim");
    assert(std::string(crisperwhisper::mode_name(
        crisperwhisper::Mode::Intended
    )) == "intended");
    assert(std::string(crisperwhisper::mode_name(
        crisperwhisper::Mode::Verbatimize
    )) == "verbatimize");

    constexpr int frames = 20;
    constexpr int mel_frames = frames * 2;
    constexpr int mel_bins = 2;
    const std::vector<int> token_ids = {100, 220, 101};
    const std::vector<std::string> pieces = {
        " hello", " ", " world",
    };
    std::vector<float> attention(
        token_ids.size() * frames, 0.001F
    );
    for (int frame = 0; frame < frames; ++frame) {
        const auto gaussian = [frame](const double center) {
            const double distance =
                static_cast<double>(frame) - center;
            return static_cast<float>(
                std::exp(-(distance * distance) / 3.0)
            );
        };
        attention[static_cast<std::size_t>(frame)] =
            gaussian(4.0);
        attention[
            static_cast<std::size_t>(frames + frame)
        ] = gaussian(9.0);
        attention[
            static_cast<std::size_t>(2 * frames + frame)
        ] = gaussian(14.0);
    }
    std::vector<float> mel(
        static_cast<std::size_t>(mel_bins * mel_frames), -1.0F
    );
    for (int bin = 0; bin < mel_bins; ++bin) {
        for (int frame = 4; frame <= 32; ++frame) {
            mel[
                static_cast<std::size_t>(bin * mel_frames + frame)
            ] = -0.2F;
        }
    }

    const auto aligned =
        crisperwhisper::detail::extract_word_timings(
            token_ids,
            pieces,
            attention,
            frames,
            mel,
            mel_bins,
            mel_frames
        );
    assert(aligned.size() == 2);
    assert(aligned[0].word == "hello");
    assert(aligned[1].word == "world");
    assert(aligned[0].start_seconds.has_value());
    assert(aligned[0].end_seconds.has_value());
    assert(aligned[1].start_seconds.has_value());
    assert(aligned[1].end_seconds.has_value());
    assert(
        *aligned[0].start_seconds <= *aligned[0].end_seconds
    );
    assert(
        *aligned[0].end_seconds <= *aligned[1].start_seconds
    );

    crisperwhisper::TranscriptionOptions default_options;
    assert(!default_options.word_timestamps);

    assert(
        crisperwhisper::detail::duplicate_prefix_words(
            {"before", "we", "love", "speech"},
            {"old", "we,", "love", "speech", "today"},
            5
        ) == 4
    );
    assert(
        crisperwhisper::detail::duplicate_prefix_words(
            {"the"}, {"the"}, 1
        ) == 0
    );
    assert(
        crisperwhisper::detail::duplicate_prefix_words(
            {"country."}, {"country"}, 1
        ) == 1
    );
    const std::vector<crisperwhisper::detail::AlignedWord>
        repeated_seam = {
            {"Americans,", 0.04, 0.08},
            {"ask", 0.12, 0.24},
            {"not", 0.26, 0.34},
            {"Americans", 12.0, 12.1},
        };
    assert(
        crisperwhisper::detail::timestamp_duplicate_prefix_words(
            repeated_seam,
            repeated_seam.size(),
            26.0,
            "Americans",
            26.08
        ) == 1
    );
    assert(
        crisperwhisper::detail::timestamp_duplicate_prefix_words(
            repeated_seam,
            repeated_seam.size(),
            26.0,
            "Americans",
            25.90
        ) == 0
    );
    assert(crisperwhisper::detail::equivalent_word("we,", "WE"));
    return 0;
}
