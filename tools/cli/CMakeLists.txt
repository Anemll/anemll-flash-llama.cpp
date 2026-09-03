set(TARGET llama-cli)
add_executable(${TARGET} cli.cpp)
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# cli.cpp reports the Flash-MoE GPU-bank build state; mirror the definition applied to libllama
if (LLAMA_FLASH_MOE_GPU_BANK)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_FLASH_MOE_GPU_BANK=1)
endif()

# cli.cpp reports the compiled HY4 IQ2 LUT path in the Flash-MoE startup block
if (LLAMA_FLASH_MOE_HY4_DIRECT_IQ2_LUT AND GGML_METAL)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_FLASH_MOE_HY4_DIRECT_IQ2_LUT=1)
endif()

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
