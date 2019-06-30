from model.u_net import get_unet

input_size = 128

max_epochs = 10
batch_size = 16

orig_width = 1918
orig_height = 1280

threshold = 0.5

model_factory = get_unet(input_size)
