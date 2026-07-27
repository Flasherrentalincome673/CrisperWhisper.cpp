#include "crisperwhisper.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string audio;
    crisperwhisper::TranscriptionOptions transcription;
    int warmup_runs = 1;
    int measured_runs = 5;
};

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

Options parse_args(int argc, char ** argv) {
    Options options;
    const unsigned int concurrency = std::thread::hardware_concurrency();
    options.transcription.threads =
        static_cast<int>(concurrency == 0 ? 4 : concurrency);

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-m" || arg == "--model") {
            options.model = next_value(i, argc, argv, arg);
        } else if (arg == "-f" || arg == "--file") {
            options.audio = next_value(i, argc, argv, arg);
        } else if (arg == "--mode") {
            const std::string mode = next_value(i, argc, argv, arg);
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
        } else if (arg == "--language") {
            options.transcription.language =
                next_value(i, argc, argv, arg);
        } else if (arg == "--threads") {
            options.transcription.threads =
                std::stoi(next_value(i, argc, argv, arg));
        } else if (arg == "--warmup") {
            options.warmup_runs =
                std::stoi(next_value(i, argc, argv, arg));
        } else if (arg == "--runs") {
            options.measured_runs =
                std::stoi(next_value(i, argc, argv, arg));
        } else if (arg == "--max-tokens") {
            options.transcription.max_new_tokens =
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
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: crisper-whisper-bench --model MODEL "
                << "--file AUDIO [--warmup 1] [--runs 5]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (options.model.empty() || options.audio.empty()) {
        throw std::invalid_argument("--model and --file are required");
    }
    if (options.warmup_runs < 0 || options.measured_runs < 1) {
        throw std::invalid_argument(
            "--warmup must be non-negative and --runs must be positive"
        );
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

struct Run {
    double call_seconds = 0.0;
    crisperwhisper::TranscriptionResult result;
};

Run transcribe_once(
    crisperwhisper::Model & model,
    const Options & options
) {
    const auto started = std::chrono::steady_clock::now();
    auto result = model.transcribe_file(
        options.audio, options.transcription
    );
    const double call_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    return Run{call_seconds, std::move(result)};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_args(argc, argv);

        const auto load_started = std::chrono::steady_clock::now();
        crisperwhisper::Model model(options.model);
        const double load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_started
        ).count();

        for (int i = 0; i < options.warmup_runs; ++i) {
            (void)transcribe_once(model, options);
        }

        std::vector<Run> runs;
        runs.reserve(static_cast<std::size_t>(options.measured_runs));
        for (int i = 0; i < options.measured_runs; ++i) {
            runs.push_back(transcribe_once(model, options));
        }

        std::cout << std::setprecision(9)
                  << "{\n"
                  << "  \"engine\": \"cpp-ggml-cuda\",\n"
                  << "  \"model_load_seconds\": " << load_seconds << ",\n"
                  << "  \"warmup_runs\": " << options.warmup_runs << ",\n"
                  << "  \"runs\": [\n";
        for (std::size_t i = 0; i < runs.size(); ++i) {
            const auto & run = runs[i];
            const auto & result = run.result;
            const double rtf =
                run.call_seconds / result.duration_seconds;
            std::cout
                << "    {\"call_seconds\": " << run.call_seconds
                << ", \"processing_seconds\": "
                << result.processing_seconds
                << ", \"duration_seconds\": "
                << result.duration_seconds
                << ", \"rtf\": " << rtf
                << ", \"chunks\": " << result.chunks.size()
                << ", \"text\": \"" << json_escape(result.text) << "\"}";
            if (i + 1 != runs.size()) {
                std::cout << ',';
            }
            std::cout << '\n';
        }
        std::cout << "  ]\n}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
