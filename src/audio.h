#pragma once

#include <string>
#include <vector>

namespace crisperwhisper::detail {

// Decode and resample supported audio to mono, 16 kHz, float PCM.
// miniaudio handles WAV, MP3 and FLAC; stb_vorbis adds Ogg Vorbis.
std::vector<float> load_audio_16khz_mono(const std::string & path);

} // namespace crisperwhisper::detail
