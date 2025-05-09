cmake_minimum_required(VERSION 3.15)
project(hlz)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python 3.7 REQUIRED COMPONENTS Interpreter Development.Module)

# Add nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(nanobind)

# Create the Python module
nanobind_add_module(hlz 
    src/lz.cpp
    src/py_bindings.cpp
)

# Enable testing
enable_testing()

# Add additional compile options and definitions
target_compile_options(hlz PRIVATE 
    -Wall 
    -Wextra
)

target_compile_definitions(hlz PRIVATE
    LZ_VERSION="${PROJECT_VERSION}"
)

# Install the module
install(TARGETS hlz
    LIBRARY DESTINATION ${Python_SITEARCH}
    RUNTIME DESTINATION ${Python_SITEARCH}
)

# Print some configuration information
message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python include: ${Python_INCLUDE_DIRS}")
message(STATUS "Project version: ${PROJECT_VERSION}") 