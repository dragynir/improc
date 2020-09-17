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
from transforms import Transforms as T


class ColorsFormats(object):
    RGB = 'rgb: {}, {}, {}'
    HSV = 'hsv: {}, {}, {}'
    LAB = 'lab: {}, {}, {}'


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.load_image()
        self.root_layout = QHBoxLayout()

        self.control_view = self.build_control_view()
        self.view_layout = self.build_view_layout()

        self.root_layout.addLayout(self.view_layout, stretch=65)
        self.root_layout.addLayout(self.control_view, stretch=35)

        self.setWindowTitle('Drag and Drop Example')
        self.setLayout(self.root_layout)
        self.show()


    def build_labeled_slider(self, text, min, max, init_value):
        h_layout = QHBoxLayout()

        slider = QSlider(orientation=Qt.Horizontal)
        slider.setRange(min, max)
        slider.setValue(init_value)
        label = QLabel(text)
        value_label = QLabel(str(init_value))

        h_layout.addWidget(label, stretch=10)
        h_layout.addWidget(slider, stretch=80)
        h_layout.addWidget(value_label, stretch=10)

        return h_layout, value_label, slider


    def build_control_view(self):


        control_view = QVBoxLayout()

        h_layout = QHBoxLayout()
        v_layout = QVBoxLayout()
        v_layout.setAlignment(Qt.AlignTop)

        s1, self.hue_slider_label, self.hue_slider = \
                self.build_labeled_slider('H: ', 0, 360, 0)

        s2, self.saturation_slider_label, self.saturation_slider = \
                self.build_labeled_slider('S: ', 0, 100, 50)

        s3, self.value_slider_label, self.value_slider = \
                self.build_labeled_slider('V: ', 0, 100, 50)

        self.hue_slider.valueChanged[int].connect(self.on_image_hsv_change)
        self.saturation_slider.valueChanged[int].connect(self.on_image_hsv_change)
        self.value_slider.valueChanged[int].connect(self.on_image_hsv_change)

        colors_info_layout = QVBoxLayout()
        self.rgb_label = QLabel(ColorsFormats.RGB.format(0, 0, 0))
        self.hsv_label = QLabel(ColorsFormats.HSV.format(0, 0, 0))
        self.lab_label = QLabel(ColorsFormats.LAB.format(0, 0, 0))
        colors_info_layout.addWidget(self.rgb_label)
        colors_info_layout.addWidget(self.hsv_label)
        colors_info_layout.addWidget(self.lab_label)

        v_layout.addLayout(s1)
        v_layout.addLayout(s2)
        v_layout.addLayout(s3)
        v_layout.addLayout(colors_info_layout)

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
        self.image_canvas.mpl_connect('motion_notify_event', self.mouse_moved_on_image)
        self.hist_figure = Figure()
        self.hist_canvas = FigureCanvas(self.hist_figure)
        
        view_layout = QVBoxLayout()

        info_layout = QHBoxLayout()

        info_layout.addWidget(self.hist_canvas, stretch=30)
        info_layout.addWidget(QWidget(), stretch=40)

        view_layout.addWidget(self.image_canvas, stretch=80)
        view_layout.addLayout(info_layout, stretch=20)
        self.show_image(self.image)

        return view_layout

    def load_image(self):
        self.image = np.array(Image.open('res\\niceimage.jpg'))
        self.shown_image = self.image

    
    def mouse_moved_on_image(self, mouse_event):
        if not mouse_event.inaxes:
            return
        
        x, y = mouse_event.xdata, mouse_event.ydata

        r, g, b = map(int, self.shown_image[int(y), int(x)])
            
        self.rgb_label.setText(ColorsFormats.RGB.format(r, g, b))

    def on_image_hsv_change(self, value):
        h = self.hue_slider.value()
        s = self.saturation_slider.value()
        v = self.value_slider.value()

        self.hue_slider_label.setText(str(h)) 
        self.saturation_slider_label.setText(str(s))
        self.value_slider_label.setText(str(v))

        s = s / 100.0 + 0.5
        v = v / 100.0 + 0.5
        self.shown_image = T.transform_hsv(self.image, h, s, v).numpy()
        self.show_image(self.shown_image)

    def show_image(self, image):
        self.image_figure.clear()
        axs = self.image_figure.add_subplot(111)
        axs.imshow(image, interpolation='none')
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
