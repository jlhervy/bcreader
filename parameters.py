HEIGHT = 118
WIDTH = 486
nb_img = 50
channels = 1
epochs = 15
generation_folder = "inference"
batch_size = 32
nb_classes = 10
max_barcode_number = 9999999999
barcode_type = 'code39'
input_shape = (HEIGHT, WIDTH, channels)
barcode_options = dict(write_text=False, module_width=0.2, module_height=8.0, quiet_zone=1.5, font_size=10, text_distance=0.0, background="white", foreground="black")

