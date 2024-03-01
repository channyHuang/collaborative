CMAKE_MINIMUM_REQUIRED(VERSION 3.8.2)

set(projectName collaborative)
project(${projectName})

find_package(Eigen3 REQUIRED)

FILE(GLOB FILES_C "*.cpp")
FILE(GLOB FILES_H "*.h" "*.inl")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)
set(OWN_DIR "" CACHE PATH "")

include_directories(${projectName}
	${EIGEN3_INCLUDE_DIR}
	)
	
link_directories(${projectName}
	
	)

add_executable(${projectName}
${FILES_C}
${FILES_H}
)

target_link_libraries(${projectName}
	
)
