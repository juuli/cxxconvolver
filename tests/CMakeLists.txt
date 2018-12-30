


SET(CMAKE_CXX_FLAGS "-frounding-math -fPIC")
set(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
#ADD INCLUDE & LINK DIRECTORIES

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/opt/local/include/eigendb/)

LINK_DIRECTORIES(${BOOST_LIBRARYDIR})
LINK_DIRECTORIES(/opt/local/lib)
LINK_DIRECTORIES(/usr/lib)

# EXECUTABLES
ADD_EXECUTABLE(ConvolverTest cxxConvolver_tests.cpp ${CPP_SOURCES} )

if(WIN32)
 set(FFTW_LIB libfftw3f)
else()
  set(FFTW_LIB fftw3f)
endif()

# LINK TARGETS
TARGET_LINK_LIBRARIES(ConvolverTest
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${FFTW_LIB})

set_target_properties( ConvolverTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ )
