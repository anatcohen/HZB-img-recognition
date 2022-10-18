import PySimpleGUI as sg
import os.path
import matplotlib as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cor
import cv2
import reg

IMG_UPLOAD_LAYOUT = '-IMAGE_LAY-'
LOAD_LAYOUT = '-LOAD_LAY-'
COR_LAYOUT = '-COR_LAY-'
REG_LAYOUT = '-REG_LAY-'

img_upload_vis = True
load_vis = cor_vis = reg_vis = False

reg_img = None

sg.LOOK_AND_FEEL_TABLE['theme'] = {'BACKGROUND': '#ffffff', 'TEXT': '#000000', 'INPUT': '#ffffff',
                                   'TEXT_INPUT': '#000000', 'SCROLL': '#ffffff', 'BUTTON': ('#ffffff', '#404040'),
                                   'PROGRESS': ('#000000', '#404040'), 'BORDER': 0, 'SLIDER_DEPTH': 0,
                                   'PROGRESS_DEPTH': 0, }
sg.theme('theme')

# Upload CL&EBSD data layout
cl_col = [
    [sg.Text("Upload CL Image")],
    [
        sg.Input(enable_events=True, key="-CL_FILE-", visible=False),
        sg.FileBrowse(size=(7, 1))
    ],
    [sg.Image(key="-CL_IMG-")]
]
ebsd_col = [
    [sg.Text("Upload EBSD Image")],
    [
        sg.Input(enable_events=True, key="-EBSD_FILE-", visible=False),
        sg.FileBrowse(size=(7, 1))
    ],
    [sg.Image(key="-EBSD_IMG-")]
]
submit_col = [
    [sg.Button(' Register', key="-REGISTER-", button_color='#4c7ab0', visible=False)],
    [sg.Button('Correlate', key='-CORRELATE-', button_color='#25a138')]
]
image_upload_layout = [
    [
        sg.Column(cl_col, element_justification="c"),
        sg.VSeperator(),
        sg.Column(ebsd_col, element_justification="c")
    ],
    [
        sg.Column(submit_col, visible=False, key='-SUBMIT-', element_justification="right", expand_x=True)
    ]
]

# Loading screen layout
loading_layout = [[sg.Text('Loading...')]]

# Correlation screen layout
cor_lay = [
    [sg.Button('', key='-BACK-', image_filename='icons/back.png', button_color='#ffffff')],
    [
        sg.Text('Colour Masks:'),
        sg.Button('  i  ', key='-COLOUR_0-', button_color='#ffbc9b', size=(4,1)),
        sg.Button('  i  ', key='-COLOUR_1-', button_color='#fffc9b', size=(4,1)),
        sg.Button('  i  ', key='-COLOUR_2-', button_color='#9eff9b', size=(4,1)),
        sg.Button('  i  ', key='-COLOUR_3-', button_color='#9bffff', size=(4,1)),
        sg.Button('  i  ', key='-COLOUR_4-', button_color='#9ba5ff', size=(4,1)),
        sg.Button('  i  ', key='-COLOUR_5-', button_color='#d49bff', size=(4,1)),
        sg.Button('  i  ', key='-COLOUR_6-', button_color='#ff9bde', size=(4,1)),
        sg.Push(),
        sg.InputText(key='-SAVE_COR-', do_not_clear=False, enable_events=True, visible=False),
        sg.FileSaveAs()
    ],
    [sg.Canvas(key="-MULT_CANVAS-")]
]

# Registration screen layout
reg_layout = [
    [sg.Button('', key='-BACK-', image_filename='icons/back.png', button_color='#ffffff')],
    [
        sg.Text('Registered Image'),
        sg.Push(),
        sg.InputText(key='-SAVE_REG-', do_not_clear=False, enable_events=True, visible=False),
        sg.FileSaveAs()
    ],
    [sg.Column([[sg.Image(key='-REG_IMG-')]], element_justification="c")],
    [sg.Text('Accuracy Tests')],
    [
        sg.Column([
            [
                sg.InputText(key='-SAVE_SUPER_IMPOSE-', do_not_clear=False, enable_events=True, visible=False),
                sg.FileSaveAs()
            ],
            [sg.Image(key='-SUPER_IMPOSE_IMG-')]

        ], element_justification="c"),
        sg.Column([
            [
                sg.InputText(key='-SAVE_INTER-', do_not_clear=False, enable_events=True, visible=False),
                sg.FileSaveAs()
            ],
            [sg.Image(key='-INTERSECT_IMG-')]
        ], element_justification="c"),
    ]
]

# General layout
layout = [
    [
        sg.Column(image_upload_layout, key=IMG_UPLOAD_LAYOUT),
        sg.Column(loading_layout, key=LOAD_LAYOUT, visible=False),
        sg.Column(cor_lay, key=COR_LAYOUT, visible=cor_vis),
        sg.Column(reg_layout, key=REG_LAYOUT, visible=reg_vis)
    ]
]

window = sg.Window('', layout)
# window.maximize()

# Enables matplotlib graphs
plt.use("TkAgg")
fig = plt.figure.Figure()


# Draws matplotlib graphs on canvas
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


while True:
    event, values = window.read()
    # Terminate window
    if event == sg.WIN_CLOSED:
        break
    # Switch to register screen
    elif event == "-REGISTER-":
        img_upload_vis = False
        reg_vis = True

        window[IMG_UPLOAD_LAYOUT].update(visible=False)
        window[REG_LAYOUT].update(visible=True)

        reg_img, intersect_img, super_impose_img = reg.get_reg_and_tests(values["-EBSD_FILE-"], values["-CL_FILE-"])

        reg_img = cv2.resize(reg_img, (400, 400), interpolation=cv2.INTER_AREA)
        reg_byte = cv2.imencode(".png", reg_img)[1].tobytes()
        window["-REG_IMG-"].update(data=reg_byte)

        intersect_img = cv2.resize(intersect_img, (400, 400), interpolation=cv2.INTER_AREA)
        intersect_byte = cv2.imencode(".png", intersect_img)[1].tobytes()
        window["-INTERSECT_IMG-"].update(data=intersect_byte)

        super_impose_img = cv2.resize(super_impose_img, (400, 400), interpolation=cv2.INTER_AREA)
        super_impose_byte = cv2.imencode(".png", super_impose_img)[1].tobytes()
        window["-SUPER_IMPOSE_IMG-"].update(data=super_impose_byte)

    # Switch to correlation screen
    elif event == "-CORRELATE-":
        img_upload_vis = False
        cor_vis = True

        window[IMG_UPLOAD_LAYOUT].update(visible=False)
        window[COR_LAYOUT].update(visible=True)

        fig = cor.get_fig(ebsd_path, cl_path)
        draw_figure(window["-MULT_CANVAS-"].TKCanvas, fig)

    # Returns to home page
    elif event == '-BACK-':
        img_upload_vis = True
        cor_vis = reg_vis = False

        window[COR_LAYOUT].update(visible=False)
        window[REG_LAYOUT].update(visible=False)
        window[IMG_UPLOAD_LAYOUT].update(visible=True)

    # Save correlation histograms
    elif event == '-SAVE_COR-' or event == '-SAVE_REG-' or event == '-SAVE_INTER-' or event == '-SAVE_SUPER_IMPOSE-':
        filename = values[event]
        if filename:
            if cor_vis:
                fig.savefig(filename)
            elif event == '-SAVE_REG-':
                cv2.imwrite(filename, reg_img)
            elif event == '-SAVE_INTER-':
                cv2.imwrite(filename, intersect_img)
            elif event == '-SAVE_SUPER_IMPOSE-':
                cv2.imwrite(filename, super_impose_img)
    elif event == '-COLOUR_0-' or event == '-COLOUR_1-' or event == '-COLOUR_2-' or event == '-COLOUR_3-' or\
            event == '-COLOUR_4-' or event == '-COLOUR_5-' or event == '-COLOUR_6-':
        ind = event[-2]
        ind_hist_lay = [
            [sg.Image(key='-MAP1-')],
            [sg.Image(key='-MAP2-')],
        ]

        temp_win = sg.Window('', ind_hist_lay, finalize=True)
        map1, map2 = cor.get_ind_figs(ebsd_path, int(ind))

        map1 = cv2.resize(map1, (400, 400), interpolation=cv2.INTER_AREA)
        map1 = cv2.imencode(".png", map1)[1].tobytes()
        temp_win["-MAP2-"].update(data=map1)

        map2 = cv2.resize(map2, (400, 400), interpolation=cv2.INTER_AREA)
        map2 = cv2.imencode(".png", map2)[1].tobytes()
        temp_win["-MAP1-"].update(data=map2)

        while True:
            temp_event, temp_values = temp_win.read()
            if temp_event == sg.WIN_CLOSED:
                break

    cl_path = values["-CL_FILE-"]
    ebsd_path = values["-EBSD_FILE-"]

    # Image upload screen
    if img_upload_vis:
        # Displays selected CL image
        if os.path.exists(cl_path):
            cl_img = cv2.imread(cl_path)
            cl_img = cv2.resize(cl_img, (400, 400), interpolation=cv2.INTER_AREA)
            cl_img = cv2.imencode(".png", cl_img)[1].tobytes()
            window["-CL_IMG-"].update(data=cl_img)
        # Displays selected EBSD image
        if os.path.exists(ebsd_path):
            ebsd_img = cv2.imread(ebsd_path)
            ebsd_img = cv2.resize(ebsd_img, (400, 400), interpolation=cv2.INTER_AREA)
            ebsd_img = cv2.imencode(".png", ebsd_img)[1].tobytes()
            window["-EBSD_IMG-"].update(data=ebsd_img)
        # Displays register button once both CL and EBSD images have been selected
        if os.path.exists(ebsd_path) and os.path.exists(cl_path):

            window['-SUBMIT-'].Update(visible=True)
