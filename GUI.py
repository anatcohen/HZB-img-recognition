import PySimpleGUI as sg
import os.path
import io
from PIL import Image

# file_types = [("JPEG (*.jpg)", "*.jpg", "*j.peg", "*.jpe"), ("JPEG 2000 (*.jpg)", "*.jp2",), ("BMP (*.bmp)", "*.bmp"),
#               ("PBM (*pbm)", "*.pbm", "*.pgm", "*.ppm"), ("SUNRASTER (*.sr)", "*.sr", "*.ras"),
#               ("TIFF (*.tiff)", "*.tiff", "*.tif"), ("PNG (*.png)", "*.png")]


# file_types = [("*.jpg", "*.jpeg", "*.jpe"), ("JPEG 2000", "*.jp2",), ("BMP", "*.bmp"),
#               ("PBM", "*.pbm", "*.pgm", "*.ppm"), ("SUNRASTER", "*.sr", "*.ras"), ("TIFF", "*.tiff", "*.tif"),
#               ("PNG", "*.png")]

IMG_UPLOAD_LAYOUT = '-IMAGE_LAY-'
COR_LAYOUT = '-COR_LAY-'
REG_LAYOUT = '-REG_LAY-'

img_upload_vis = True
cor_vis = False
reg_vis = True

cl_col = [
    [sg.Text("Upload CL Image")],
    [
        sg.Input(size=(25, 1), enable_events=True, key="-CL_FILE-"),
        sg.FileBrowse()
    ],
    [sg.Image(key="-CL_IMG-")]
]

ebsd_col = [
    [sg.Text("Upload EBSD Image:")],
    [
        sg.Input(size=(25, 1), enable_events=True, key="-EBSD_FILE-"),
        sg.FileBrowse()
    ],
    [sg.Image(key="-EBSD_IMG-")]
]

submit_col = [[sg.Button('Register', key="-REGISTER-")], [sg.Button('Correlate', key='-CORRELATE-')]]

image_upload_layout = [
    [
        # sg.Column(image_load, key=image_load)
        sg.Column(cl_col, key='-CL_COL-'),
        sg.VSeperator(),
        sg.Column(ebsd_col),
        sg.Column(submit_col, visible=False, key='-SUBMIT-')
    ]
]

cor_layout = [[sg.Text('Test')]]

reg_layout = [[]]

layout = [
    [
        sg.Column(image_upload_layout, key=IMG_UPLOAD_LAYOUT),
        sg.Column(cor_layout, key=COR_LAYOUT, visible=False),
        sg.Column(cor_layout, key=REG_LAYOUT, visible=False)
    ]
]

window = sg.Window(IMG_UPLOAD_LAYOUT, layout)

while True:
    event, values = window.read()
    # Terminate window
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == "Register":
        print('done')
    elif event == "-CORRELATE-":
        # window = sg.Window(CORRELATION_LAYOUT, cor_layout)
        window[IMG_UPLOAD_LAYOUT].update(visible=False)
        img_upload_vis = False
        cor_vis = True
        print( window[IMG_UPLOAD_LAYOUT].)
        window[COR_LAYOUT].update(visible=True)
    cl_path = values["-CL_FILE-"]
    ebsd_path = values["-EBSD_FILE-"]

    # Displays selected CL image
    if img_upload_vis:
        if os.path.exists(cl_path):
            image = Image.open(cl_path)
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-CL_IMG-"].update(data=bio.getvalue())
        # Displays selected EBSD image
        if os.path.exists(ebsd_path):
            image = Image.open(ebsd_path)
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-EBSD_IMG-"].update(data=bio.getvalue())
        # Displays register button once both CL and EBSD images have been selected
        if os.path.exists(ebsd_path) and os.path.exists(cl_path):
            window['-SUBMIT-'].Update(visible=True)


