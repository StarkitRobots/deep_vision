
# Add executable called "helloDemo" that is built from the source files
# "demo.cxx" and "demo_b.cxx". The extensions are automatically found.
add_executable (ball_test ball_test.cpp) #broken with tiny-dnn update?
add_executable (ball_train ball_train.cpp)
add_executable (ball_train_cifar ball_train_cifar.cpp)
add_executable (ball_test_cifar ball_test_cifar.cpp)
add_executable (ball_slidingdetect ball_slidingdetect.cpp)
add_executable (ball_train_exp ball_train_exp.cpp)
add_executable (ball_test_exp ball_test_exp.cpp)
add_executable (predict predict.cpp)
add_executable (dump_detected dump_detected.cpp)
add_executable (test_prediction test_prediction.cpp)
add_executable (post_train_exp post_train_exp.cpp)
add_executable (analyze_acceptance_score analyze_acceptance_score.cpp)

# Link the executable to the Hello library. Since the Hello library has
# public include directories we will use those link directories when building
# helloDemo
# target_link_libraries (helloDemo LINK_PUBLIC Hello)
#target_link_libraries( ball_train ${CMAKE_THREAD_LIBS_INIT} )
#target_link_libraries( ball_test ${CMAKE_THREAD_LIBS_INIT} )

target_link_libraries(ball_test_cifar ${OpenCV_LIBS} )
target_link_libraries(ball_slidingdetect ${OpenCV_LIBS} )
target_link_libraries(predict ${OpenCV_LIBS} )
target_link_libraries(ball_test_exp ${OpenCV_LIBS} )
target_link_libraries(dump_detected ${OpenCV_LIBS} )
target_link_libraries(test_prediction ${OpenCV_LIBS} )
target_link_libraries(post_train_exp ${OpenCV_LIBS} )
target_link_libraries(analyze_acceptance_score ${OpenCV_LIBS} )

# add_executable(benchmarks_all benchmarks/main.cpp ${tiny_dnn_headers})
# target_link_libraries(benchmarks_all
#     ${project_library_target_name} ${REQUIRED_LIBRARIES})
