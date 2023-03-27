# Set a default build type if none was specified
set(default_build_type "Release")

#COMMENT OUT to default to using a debug build if the source dir is a git clone
#if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
#  set(default_build_type "Debug")
#endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()
