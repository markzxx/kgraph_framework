set(CMAKE_CXX_STANDARD 11)

#add_executable(test_nndescent test_nndescent.cpp)
#target_link_libraries(test_nndescent ${PROJECT_NAME})
#
#add_executable(test_nndescent_refine test_nndescent_refine.cpp)
#target_link_libraries(test_nndescent_refine ${PROJECT_NAME})
#
#add_executable(test_kdtree_graph test_kdtree_graph.cpp)
#target_link_libraries(test_kdtree_graph ${PROJECT_NAME})

add_executable(kdtree kdtree_buildall_and_evaluate.cpp)
if (WIN32)
    target_link_libraries(kdtree ${PROJECT_NAME})
else ()
    target_link_libraries(kdtree ${PROJECT_NAME} -lprofiler -lmysqlclient)
endif ()

add_executable(random random_buildall_and_evaluate.cpp)
target_link_libraries(random ${PROJECT_NAME} -lmysqlclient)

add_executable(bruteforce bruteforce.cpp)
target_link_libraries(bruteforce ${PROJECT_NAME} -lmysqlclient)

add_executable(clsh rplsh_buildall_and_evaluate.cpp)
if (WIN32)
    target_link_libraries(clsh ${PROJECT_NAME} -lopenblas)
else ()
    target_link_libraries(clsh ${PROJECT_NAME} -lprofiler -lopenblas -lmysqlclient)
endif ()