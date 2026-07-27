#include "crisperwhisper.h"

#include <cassert>
#include <string>

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
    return 0;
}
