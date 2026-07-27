#include "audio.h"

#include <stdexcept>
#include <string>
#include <vector>

#define STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"

#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

namespace crisperwhisper::detail {

std::vector<float> load_audio_16khz_mono(const std::string & path) {
    const ma_decoder_config config =
        ma_decoder_config_init(ma_format_f32, 1, 16'000);

    ma_decoder decoder{};
    const ma_result init_result =
        ma_decoder_init_file(path.c_str(), &config, &decoder);
    if (init_result != MA_SUCCESS) {
        throw std::runtime_error(
            "could not decode audio '" + path + "': " +
            ma_result_description(init_result)
        );
    }

    struct DecoderGuard {
        ma_decoder * decoder;
        ~DecoderGuard() {
            ma_decoder_uninit(decoder);
        }
    } guard{&decoder};

    ma_uint64 frame_count = 0;
    const ma_result length_result =
        ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (length_result != MA_SUCCESS) {
        throw std::runtime_error(
            "could not determine audio length: " +
            std::string(ma_result_description(length_result))
        );
    }
    if (frame_count == 0) {
        throw std::runtime_error("audio file contains no samples: " + path);
    }

    std::vector<float> samples(static_cast<std::size_t>(frame_count));
    ma_uint64 frames_read = 0;
    const ma_result read_result = ma_decoder_read_pcm_frames(
        &decoder, samples.data(), frame_count, &frames_read
    );
    if (read_result != MA_SUCCESS) {
        throw std::runtime_error(
            "could not read audio samples: " +
            std::string(ma_result_description(read_result))
        );
    }
    samples.resize(static_cast<std::size_t>(frames_read));
    return samples;
}

} // namespace crisperwhisper::detail

#undef STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"
