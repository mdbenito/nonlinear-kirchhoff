cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME nonlinear_kirchhoff)
project(${PROJECT_NAME})

cmake_policy(SET CMP0004 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # MBD: For rtags

##############################################################################
# Get DOLFIN configuration data
# DOLFINConfig.cmake must be in DOLFIN_CMAKE_CONFIG_PATH

find_package(DOLFIN REQUIRED)

if (EXISTS ${DOLFIN_USE_FILE})
  message(STATUS "Found DOLFIN_USE_FILE = " ${DOLFIN_USE_FILE})
  include(${DOLFIN_USE_FILE})

  # Default build type (can be overridden by user)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
  endif()
  message(STATUS "Build type is " ${CMAKE_BUILD_TYPE})

  # Do not throw error for 'multi-line comments' (these are typical in
  # rst which includes LaTeX)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
  if (HAVE_NO_MULTLINE)
    set(CMAKE_CXX_FLAGS "-Wno-comment ${CMAKE_CXX_FLAGS}")
  endif()

else()
  # Compiler definitions
  add_definitions(${DOLFIN_CXX_DEFINITIONS})

  # Compiler flags
  set(CMAKE_CXX_FLAGS "${DOLFIN_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

  # Do not throw error for 'multi-line comments' (these are typical in
  # rst which includes LaTeX)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
  if (HAVE_NO_MULTLINE)
    set(CMAKE_CXX_FLAGS "-Wno-comment ${CMAKE_CXX_FLAGS}")
  endif()

  # Include directories
  include_directories(${DOLFIN_INCLUDE_DIRS})
  include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})
endif()

##############################################################################
# Compile UFL files with FFC. Adapted from:
# https://bitbucket.org/fenics-project/dolfin/src/efc41658e5e10bf16fb02489c57e20e3c291fade/CMakeLists.txt
# TODO: enable optimization once it is implemented in FFC

set(UFL_FILENAMES "${CMAKE_SOURCE_DIR}/NonlinearKirchhoff.ufl")


# COMMAND escapes whitespace so we pass a list which it automatically expands
set(FFC_FLAGS "-l" "dolfin" "-r" "quadrature" "-O0" "-fsplit") 
foreach(UFL_FILENAME ${UFL_FILENAMES})
  get_filename_component(UFL_BASENAME ${UFL_FILENAME} NAME_WE)
  get_filename_component(UFL_DIRECTORY ${UFL_FILENAME} DIRECTORY)
  set(FFC_OUTPUT_DIR ${UFL_DIRECTORY})
  message(STATUS "Running FEniCS Form Compiler on " ${UFL_FILENAME})
  add_custom_command(
    OUTPUT ${FFC_OUTPUT_DIR}/${UFL_BASENAME}.h
           ${FFC_OUTPUT_DIR}/${UFL_BASENAME}.cpp
    MAIN_DEPENDENCY ${UFL_FILENAME}
    WORKING_DIRECTORY ${UFL_DIRECTORY}
    COMMENT Running FEniCS Form Compiler on ${UFL_FILENAME}.
    if (EXISTS "$ENV{HOME}/fenics.env.conf")
      COMMAND . $ENV{HOME}/fenics.env.conf && ffc ${FFC_FLAGS} --output-directory ${FFC_OUTPUT_DIR} ${UFL_FILENAME}
    else()
      COMMAND ffc ${FFC_FLAGS} --output-directory ${FFC_OUTPUT_DIR} ${UFL_FILENAME}
    endif()
    )

  # Accumulate list of generated sources to compile into executable
  set(FFC_GENERATED_HEADERS ${FFC_GENERATED_HEADERS}
                            ${FFC_OUTPUT_DIR}/${UFL_BASENAME}.h)
  set(FFC_GENERATED_SOURCES ${FFC_GENERATED_SOURCES}
                            ${FFC_OUTPUT_DIR}/${UFL_BASENAME}.cpp)
endforeach(UFL_FILENAME)


##############################################################################
# Executable and libraries

add_executable(${PROJECT_NAME} ${FFC_GENERATED_SOURCES}
  IsometryConstraint.cpp
  HermiteDirichletBC.cpp
  KirchhoffAssembler.cpp
  DKTGradient.cpp
  BlockMatrixAdapter.cpp
  BlockVectorAdapter.cpp
  output.cpp
  dkt_utils.cpp
  tests.cpp
  main.cpp)

target_link_libraries(${PROJECT_NAME} ${DOLFIN_LIBRARIES})
