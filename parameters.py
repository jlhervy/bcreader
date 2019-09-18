HEIGHT = 118
WIDTH = 524
nb_img = 5000
channels = 1
epochs = 25
batch_size = 64
nb_classes = 10
max_barcode_number = 9999999999
barcode_type = 'code39'
input_shape = (HEIGHT, WIDTH, channels)
barcode_options = dict(write_text=False, module_width=0.2, module_height=8.0, quiet_zone=1.5, font_size=10, text_distance=0.0, background="white", foreground="black")

