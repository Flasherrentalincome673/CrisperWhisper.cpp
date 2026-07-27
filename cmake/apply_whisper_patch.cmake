if(NOT DEFINED WHISPER_SOURCE_DIR OR NOT DEFINED WHISPER_PATCH_FILE)
    message(FATAL_ERROR "WHISPER_SOURCE_DIR and WHISPER_PATCH_FILE are required")
endif()

execute_process(
    COMMAND git apply --check "${WHISPER_PATCH_FILE}"
    WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
    RESULT_VARIABLE _can_apply
    OUTPUT_QUIET
    ERROR_QUIET
)

if(_can_apply EQUAL 0)
    execute_process(
        COMMAND git apply "${WHISPER_PATCH_FILE}"
        WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
        RESULT_VARIABLE _apply_result
    )
    if(NOT _apply_result EQUAL 0)
        message(FATAL_ERROR "Could not apply the whisper.cpp attention patch")
    endif()
else()
    execute_process(
        COMMAND git apply --reverse --check "${WHISPER_PATCH_FILE}"
        WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
        RESULT_VARIABLE _already_applied
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT _already_applied EQUAL 0)
        message(FATAL_ERROR
            "The pinned whisper.cpp source does not match the attention patch"
        )
    endif()
endif()
