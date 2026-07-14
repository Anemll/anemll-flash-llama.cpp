set(TARGET llama-flashmoe-state-probe)
add_executable(${TARGET} flashmoe-state-probe.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
