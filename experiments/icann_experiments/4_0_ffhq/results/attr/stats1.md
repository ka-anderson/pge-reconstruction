|    | model_name   | input_data         |     f_MSE |      i_MSE |     i_SSIM |    i_FaceNet |
|---:|:-------------|:-------------------|----------:|-----------:|-----------:|-------------:|
|  0 | mse          | training_img_input |   7.90499 |   0.35269  |   0.118125 |   0.00371132 |
|  1 | mse          | test_img_input     |   7.82796 |   0.353537 |   0.12001  |   0.00370835 |
|  2 | mse          | noise_input        | nan       | nan        | nan        | nan          |
|  3 | fn           | training_img_input |   4.63193 |   0.405564 |   0.119381 |   0.00358313 |
|  4 | fn           | test_img_input     |   4.57616 |   0.40166  |   0.124698 |   0.00357858 |
|  5 | fn           | noise_input        | nan       | nan        | nan        | nan          |
|  6 | mix          | training_img_input |   3.42988 |   0.342664 |   0.12745  |   0.00360893 |
|  7 | mix          | test_img_input     |   3.39144 |   0.341668 |   0.130194 |   0.00360061 |
|  8 | mix          | noise_input        | nan       | nan        | nan        | nan          |