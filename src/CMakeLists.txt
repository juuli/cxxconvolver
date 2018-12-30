


SET(CMAKE_CXX_FLAGS "-frounding-math -fPIC")
set(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
#ADD INCLUDE & LINK DIRECTORIES

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/opt/local/include/eigendb/)

LINK_DIRECTORIES(${BOOST_LIBRARYDIR})
LINK_DIRECTORIES(/opt/local/lib)
LINK_DIRECTORIES(/usr/lib)

# EXECUTABLES
ADD_EXECUTABLE(ConvolverMain main.cpp ${CPP_SOURCES} )
ADD_LIBRARY( CxxConvolver SHARED ${CPP_SOURCES} )
ADD_EXECUTABLE(PlaybackMain playback_main.cpp ${CPP_SOURCES} )

if(WIN32)
 set(FFTW_LIB libfftw3f)
 set(PA_LIB portaudio)
else()
  set(FFTW_LIB fftw3f)
  set(PA_LIB libportaudio.a)
endif()

# LINK TARGETS
TARGET_LINK_LIBRARIES(ConvolverMain
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${FFTW_LIB})

TARGET_LINK_LIBRARIES(CxxConvolver
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${FFTW_LIB})

TARGET_LINK_LIBRARIES(PlaybackMain
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${FFTW_LIB} ${PA_LIB})


set_target_properties( PlaybackMain PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ )
set_target_properties( CxxConvolver PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/ )
set_target_properties( CxxConvolver PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/ )
set_target_properties( CxxConvolver PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/ )
set_target_properties( ConvolverMain PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ )
