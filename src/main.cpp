#include "crisperwhisper.h"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct CliOptions {
    std::string model;
    std::string audio;
    crisperwhisper::ModelOptions model_options;
    crisperwhisper::TranscriptionOptions transcription;
    bool json = false;
};

[[noreturn]] void usage(const char * program, int exit_code) {
    std::ostream & out = exit_code == 0 ? std::cout : std::cerr;
    out
        << "Native CrisperWhisper 2.0 transcription\n\n"
        << "Usage:\n  " << program
        << " --model MODEL.bin --file AUDIO [options]\n\n"
        << "Required:\n"
        << "  -m, --model PATH          Converted ggml model\n"
        << "  -f, --file PATH           Audio; auto downmix/resample to 16 kHz mono\n\n"
        << "Recognition:\n"
        << "  --mode MODE               verbatim (default) or intended\n"
        << "  -l, --language CODE       Whisper language code (default: en)\n"
        << "  --hotwords TEXT           Comma-separated hotwords (Pro models)\n"
        << "  --verbatimize TEXT        Insert spoken disfluencies into TEXT\n"
        << "  --verbatize TEXT          Alias for --verbatimize\n"
        << "  --max-tokens N            Tokens per chunk (default: 256)\n"
        << "  --word-timestamps         Supervised per-word timestamps\n"
        << "  -t, --threads N           CPU worker threads\n\n"
        << "Long audio:\n"
        << "  --chunk-seconds N         Window length, max 30 (default: 30)\n"
        << "  --stride-seconds N        Window advance (default: 26)\n"
        << "  --context-words N         Continuation words (default: 12)\n"
        << "  --drop-words N            Boundary words to re-cover (default: 2)\n\n"
        << "Hardware/output:\n"
        << "  --cpu                      Disable GPU offload\n"
        << "  --gpu N                    CUDA device index (default: 0)\n"
        << "  --no-flash-attn            Disable flash attention\n"
        << "  --json                     Emit a JSON result\n"
        << "  -h, --help                 Show this help\n";
    std::exit(exit_code);
}

std::string next_value(
    int & index,
    int argc,
    char ** argv,
    const std::string & option
) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(option + " requires a value");
    }
    return argv[++index];
}

std::vector<std::string> split_commas(const std::string & value) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto begin = item.find_first_not_of(" \t");
        const auto end = item.find_last_not_of(" \t");
        if (begin != std::string::npos) {
            result.push_back(item.substr(begin, end - begin + 1));
        }
    }
    return result;
}

CliOptions parse_args(int argc, char ** argv) {
    CliOptions options;
    const unsigned int concurrency = std::thread::hardware_concurrency();
    options.transcription.threads =
        static_cast<int>(concurrency == 0 ? 4 : concurrency);

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            usage(argv[0], 0);
        } else if (arg == "-m" || arg == "--model") {
            options.model = next_value(i, argc, argv, arg);
        } else if (arg == "-f" || arg == "--file") {
            options.audio = next_value(i, argc, argv, arg);
        } else if (arg == "--mode") {
            const auto mode = next_value(i, argc, argv, arg);
            if (mode == "verbatim") {
                options.transcription.mode =
                    crisperwhisper::Mode::Verbatim;
            } else if (mode == "intended") {
                options.transcription.mode =
                    crisperwhisper::Mode::Intended;
            } else {
                throw std::invalid_argument(
                    "--mode must be verbatim or intended"
                );
            }
        } else if (arg == "-l" || arg == "--language") {
            options.transcription.language =
                next_value(i, argc, argv, arg);
        } else if (arg == "--hotwords") {
            options.transcription.hotwords = split_commas(
                next_value(i, argc, argv, arg)
            );
        } else if (arg == "--verbatimize" || arg == "--verbatize") {
            options.transcription.verbatimize_transcript =
                next_value(i, argc, argv, arg);
            options.transcription.mode =
                crisperwhisper::Mode::Verbatimize;
        } else if (arg == "--max-tokens") {
            options.transcription.max_new_tokens =
                std::stoi(next_value(i, argc, argv, arg));
        } else if (arg == "--word-timestamps") {
            options.transcription.word_timestamps = true;
        } else if (arg == "-t" || arg == "--threads") {
            options.transcription.threads =
                std::stoi(next_value(i, argc, argv, arg));
        } else if (arg == "--chunk-seconds") {
            options.transcription.chunk_seconds =
                std::stod(next_value(i, argc, argv, arg));
        } else if (arg == "--stride-seconds") {
            options.transcription.stride_seconds =
                std::stod(next_value(i, argc, argv, arg));
        } else if (arg == "--context-words") {
            options.transcription.context_words =
                std::stoi(next_value(i, argc, argv, arg));
        } else if (arg == "--drop-words") {
            options.transcription.drop_words =
                std::stoi(next_value(i, argc, argv, arg));
        } else if (arg == "--cpu") {
            options.model_options.use_gpu = false;
        } else if (arg == "--gpu") {
            options.model_options.gpu_device =
                std::stoi(next_value(i, argc, argv, arg));
        } else if (arg == "--no-flash-attn") {
            options.model_options.flash_attention = false;
        } else if (arg == "--json") {
            options.json = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (options.model.empty() || options.audio.empty()) {
        usage(argv[0], 2);
    }
    return options;
}

std::string json_escape(const std::string & value) {
    std::ostringstream output;
    for (const unsigned char c : value) {
        switch (c) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (c < 0x20) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<int>(c)
                       << std::dec;
            } else {
                output << static_cast<char>(c);
            }
        }
    }
    return output.str();
}

std::string format_timestamp(const double seconds) {
    const auto total_milliseconds = static_cast<long long>(
        std::llround(std::max(0.0, seconds) * 1000.0)
    );
    const auto hours = total_milliseconds / 3'600'000;
    const auto minutes = (total_milliseconds / 60'000) % 60;
    const auto whole_seconds = (total_milliseconds / 1'000) % 60;
    const auto milliseconds = total_milliseconds % 1'000;

    std::ostringstream output;
    output << std::setfill('0')
           << std::setw(2) << hours << ':'
           << std::setw(2) << minutes << ':'
           << std::setw(2) << whole_seconds << '.'
           << std::setw(3) << milliseconds;
    return output.str();
}

void print_json_words(
    const std::vector<crisperwhisper::WordTimestamp> & words,
    const std::string & indentation
) {
    std::cout << "[";
    if (!words.empty()) {
        std::cout << '\n';
    }
    for (std::size_t i = 0; i < words.size(); ++i) {
        const auto & word = words[i];
        std::cout
            << indentation << "  {\"word\": \"" << json_escape(word.word)
            << "\", \"start\": " << word.start_seconds
            << ", \"end\": " << word.end_seconds << "}";
        if (i + 1 != words.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }
    if (!words.empty()) {
        std::cout << indentation;
    }
    std::cout << ']';
}

void print_json(const crisperwhisper::TranscriptionResult & result) {
    std::cout << "{\n"
              << "  \"text\": \"" << json_escape(result.text) << "\",\n"
              << "  \"language\": \"" << json_escape(result.language)
              << "\",\n"
              << "  \"mode\": \"" << crisperwhisper::mode_name(result.mode)
              << "\",\n"
              << "  \"duration\": " << result.duration_seconds << ",\n"
              << "  \"processing_time\": " << result.processing_seconds
              << ",\n";
    if (result.word_timestamps) {
        std::cout << "  \"words\": ";
        print_json_words(result.words, "  ");
        std::cout << ",\n";
    }
    std::cout << "  \"chunks\": [\n";
    for (std::size_t i = 0; i < result.chunks.size(); ++i) {
        const auto & chunk = result.chunks[i];
        std::cout
            << "    {\"index\": " << chunk.index
            << ", \"start\": " << chunk.start_seconds
            << ", \"end\": " << chunk.end_seconds
            << ", \"text\": \"" << json_escape(chunk.text)
            << "\", \"context\": \"" << json_escape(chunk.context)
            << "\", \"is_last\": "
            << (chunk.is_last ? "true" : "false");
        if (result.word_timestamps) {
            std::cout << ", \"words\": ";
            print_json_words(chunk.words, "    ");
        }
        std::cout << "}";
        if (i + 1 != result.chunks.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }
    std::cout << "  ]\n}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);
        crisperwhisper::Model model(options.model, options.model_options);
        const auto result =
            model.transcribe_file(options.audio, options.transcription);

        if (options.json) {
            print_json(result);
        } else {
            std::cout << result.text << '\n';
            if (options.transcription.word_timestamps) {
                std::cout << "\nWord timestamps:\n";
                for (const auto & word : result.words) {
                    std::cout
                        << '[' << format_timestamp(word.start_seconds)
                        << " --> " << format_timestamp(word.end_seconds)
                        << "] " << word.word << '\n';
                }
            }
            std::cerr
                << "processed " << std::fixed << std::setprecision(2)
                << result.duration_seconds << " s of audio in "
                << result.processing_seconds << " s ("
                << crisperwhisper::mode_name(result.mode) << ")\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
