# script to run cats and dogs classifier
# to be used with tfruns
# Julin Maloof
# Dec 6, 2021
# Copied from listing 5.13 and 5.14 in Deep Learning with R

basedir <- "/kaggle/input/cats-dogs-small/cats_and_dogs_small"

train_dir <- file.path(basedir, "train")
validation_dir <- file.path(basedir, "validation")

# Generators:

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 10,
  class_mode = "binary"
)

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = FLAGS$filters, kernel_size = c(FLAGS$kernel, FLAGS$kernel), activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = FLAGS$filters*2, kernel_size = c(FLAGS$kernel, FLAGS$kernel), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = FLAGS$filters*2*2, kernel_size = c(FLAGS$kernel, FLAGS$kernel), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2))

if(FLAGS$four_layer) {
  model <- model %>%
    layer_conv_2d(filters = FLAGS$filters*2*2, kernel_size = c(FLAGS$kernel, FLAGS$kernel), activation = "relu") %>% 
    layer_max_pooling_2d(pool_size = c(2, 2))
}

model <- model %>% 
  layer_flatten() %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")  

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 50,
  validation_data = validation_generator,
  validation_steps = 50
)