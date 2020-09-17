from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QApplication, QWidget
from PyQt5.QtWidgets import QLabel, QSlider, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import sys
from PIL import Image


class ColorsFormats(object):
    RGB = 'rgb: {}, {}, {}'
    HSV = 'hsv: {}, {}, {}'
    LAB = 'lab: {}, {}, {}'


class Window(QWidget):

    def __init__(self):
        super().__init__()
    
        self.root_layout = QHBoxLayout()

        self.view_layout = self.build_view_layout()
        self.control_view = self.build_control_view()

        self.root_layout.addLayout(self.view_layout, stretch=65)
        self.root_layout.addLayout(self.control_view, stretch=35)

        self.setWindowTitle('Drag and Drop Example')
        self.setLayout(self.root_layout)
        self.show()


    def build_labeled_slider(self, text, min, max):
        h_layout = QHBoxLayout()

        slider = QSlider(orientation=Qt.Horizontal)
        slider.setRange(min, max)
        label = QLabel(text)

        h_layout.addWidget(label, stretch=10)
        h_layout.addWidget(slider, stretch=90)

        return h_layout, slider


    def build_control_view(self):


        control_view = QVBoxLayout()

        h_layout = QHBoxLayout()
        v_layout = QVBoxLayout()

        s1, self.hue_slider = self.build_labeled_slider('H: ', 0, 360)
        s2, self.suturation_slider = self.build_labeled_slider('S: ', 0, 100)
        s3, self.value_slider = self.build_labeled_slider('V: ', 0, 100)
        v_layout.addLayout(s1)
        v_layout.addLayout(s2)
        v_layout.addLayout(s3)

        self.operations_list_widget = QListWidget()

        h_layout.addLayout(v_layout, stretch=50)
        h_layout.addWidget(self.operations_list_widget, stretch=50)


        h_pipeline = QHBoxLayout()
        buttons_layout = QVBoxLayout()
        buttons_layout.setAlignment(Qt.AlignTop)
        
        self.run_button = QPushButton('run')
        self.open_image_button = QPushButton('open')
        buttons_layout.addWidget(self.run_button, stretch=5)
        buttons_layout.addWidget(self.open_image_button, stretch=5)


        self.pipeline_operations_widget = QListWidget()
        h_pipeline.addWidget(self.pipeline_operations_widget)
        h_pipeline.addLayout(buttons_layout)

        control_view.addLayout(h_layout, stretch=30)
        control_view.addLayout(h_pipeline, stretch=70)

        return control_view


    
    def build_view_layout(self):
        self.image_figure = Figure()
        self.image_canvas = FigureCanvas(self.image_figure)
        self.hist_figure = Figure()
        self.hist_canvas = FigureCanvas(self.hist_figure)
        
        view_layout = QVBoxLayout()

        info_layout = QHBoxLayout()

        colors_info_layout = QVBoxLayout()
        self.rgb_label = QLabel(ColorsFormats.RGB.format(0, 0, 0))
        self.hsv_label = QLabel(ColorsFormats.HSV.format(0, 0, 0))
        self.lab_label = QLabel(ColorsFormats.LAB.format(0, 0, 0))
        colors_info_layout.addWidget(self.rgb_label)
        colors_info_layout.addWidget(self.hsv_label)
        colors_info_layout.addWidget(self.lab_label)

        info_layout.addLayout(colors_info_layout, stretch=30)
        info_layout.addWidget(self.hist_canvas, stretch=30)
        info_layout.addWidget(QWidget(), stretch=40)

        view_layout.addWidget(self.image_canvas, stretch=80)
        view_layout.addLayout(info_layout, stretch=20)

        self.load_image()
        self.show_image(self.image)

        return view_layout



    def load_image(self):
        self.image = np.array(Image.open('res\\niceimage.jpg'))

    def show_image(self, image):
        self.image_figure.clear()
        axs = self.image_figure.add_subplot(111)
        axs.imshow(image)
        self.image_canvas.draw()
        


app = QApplication([])

# appStyle = """
#         QLabel { 
#             border-radius: 3px;
#             }

#         QSlider::groove:horizontal {
#                 border: 1px solid #565a5e;
#                 height: 10px;
#                 border-radius: 4px;
#             }
#         QSlider::handle:horizontal {
#                 background: green;
#                 border: 1px solid #565a5e;
#                 width: 12px;
#                 height: 24px;
#                 border-radius: 4px;
#             }

#         """

# app.setStyleSheet(appStyle)

application = Window()
application.show()
sys.exit(app.exec())
