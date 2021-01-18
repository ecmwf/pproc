cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

find_package(ecbuild 3.4 REQUIRED)

project(cmir VERSION 0.1 LANGUAGES C CXX)

find_package(eckit REQUIRED)
find_package(mir REQUIRED)

ecbuild_add_library(
    TARGET cmir
    SOURCES cmir.cc config.cc cmir.h
    INSTALL_HEADERS LISTED
    PRIVATE_LIBS eckit mir
)

ecbuild_install_project(NAME ${PROJECT_NAME})
