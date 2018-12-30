CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(CpuConvolver)
set( BOOST_ROOT "C:/Users/jks/Documents/boost_1_63_0")
set( CMAKE_BUILD_TYPE release)

if(WIN32)
  set( Boost_USE_STATIC_LIBS ON )
  set( Boost_USE_MULTITHREADED ON )
  set( Boost_USE_STATIC_RUNTIME OFF )
  set( BOOST_LIBRARYDIR ${BOOST_ROOT}/stage/lib)
endif()

FIND_PACKAGE(Boost COMPONENTS system thread filesystem unit_test_framework REQUIRED)

message( STATUS ${Boost_THREAD_LIBRARY})

set(CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/cxxConvolver.cpp)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tests)
