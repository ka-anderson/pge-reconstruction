|    | model_name   | input_data         |      f_MSE |      i_MSE |      i_SSIM |
|---:|:-------------|:-------------------|-----------:|-----------:|------------:|
|  0 | mse          | training_img_input |   0.225276 |   0.442245 |   0.087791  |
|  1 | mse          | test_img_input     |   0.205866 |   0.438721 |   0.0934136 |
|  2 | mse          | noise_input        | nan        | nan        | nan         |
|  3 | fn           | training_img_input |   0.203036 |   0.443347 |   0.0953042 |
|  4 | fn           | test_img_input     |   0.175857 |   0.439935 |   0.101571  |
|  5 | fn           | noise_input        | nan        | nan        | nan         |
|  6 | mse+fn       | training_img_input |   0.198511 |   0.432119 |   0.0912615 |
|  7 | mse+fn       | test_img_input     |   0.174383 |   0.430216 |   0.0971845 |
|  8 | mse+fn       | noise_input        | nan        | nan        | nan         |