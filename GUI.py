import PySimpleGUI as sg
import os.path
import io
from PIL import Image
import matplotlib as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cor

IMG_UPLOAD_LAYOUT = '-IMAGE_LAY-'
LOAD_LAYOUT = '-LOAD_LAY-'
COR_LAYOUT = '-COR_LAY-'
REG_LAYOUT = '-REG_LAY-'

img_upload_vis = True
load_vis = False
cor_vis = False
reg_vis = False

# Upload CL&EBSD data layout
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

# Loading screen layout
loading_layout = [[sg.Text('Loading...')]]

# Correlation screen layout
cor_lay = [
    [sg.InputText(key='-SAVE_AS-', do_not_clear=False, enable_events=True, visible=False), sg.FileSaveAs()],
    [sg.Canvas(key="-CANVAS-")]
]

# Registration screen layout
reg_layout = [[]]

# General layout
layout = [
    [
        sg.Column(image_upload_layout, key=IMG_UPLOAD_LAYOUT),
        sg.Column(loading_layout, key=LOAD_LAYOUT, visible=False),
        sg.Column(cor_lay, key=COR_LAYOUT, visible=False),
        sg.Column(reg_layout, key=REG_LAYOUT, visible=False)
    ]
]
window = sg.Window(IMG_UPLOAD_LAYOUT, layout)

# Enables matplotlib graphs
plt.use("TkAgg")
fig = plt.figure.Figure(figsize=(5, 4), dpi=100)


# Draws matplotlib graphs on canvas
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


while True:
    event, values = window.read()
    # Terminate window
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    # Switch to register screen
    elif event == "Register":
        print('done')
    # Switch to correlation screen
    elif event == "-CORRELATE-":
        window[IMG_UPLOAD_LAYOUT].update(visible=False)
        img_upload_vis = False
        cor_vis = True
        window[COR_LAYOUT].update(visible=True)
        fig = cor.get_fig(ebsd_path, cl_path)
        draw_figure(window["-CANVAS-"].TKCanvas, fig)
    # Save correlation histograms
    elif event == '-SAVE_AS-':
        filename = values['-SAVE_AS-']
        if filename:
            fig.savefig(filename)

    cl_path = values["-CL_FILE-"]
    ebsd_path = values["-EBSD_FILE-"]

    # Image upload screen
    if img_upload_vis:
        # Displays selected CL image
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
