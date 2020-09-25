from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QApplication, QWidget
from PyQt5.QtWidgets import QLabel, QSlider, QPushButton, QFileDialog, QDialog
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



class GaborDialog(QDialog):
    def __init__(self, theta):
        QDialog.__init__(self, None)

        self.theta = theta

        v_layout = QVBoxLayout()

        layout, self.label, self.angle_slider = \
                Window.build_labeled_slider('Theta: ', 0, 180, 0)
        
        self.angle_slider.valueChanged[int].connect(self.update)

        self.angle_slider.setValue(self.theta)


        v_layout.addLayout(layout)

        confirm_button = QPushButton('confirm')
        confirm_button.clicked.connect(self.on_confirm)
        v_layout.addWidget(confirm_button)
        self.setLayout(v_layout)
    
    def update(self, value):
        self.label.setText(str(value))

    def on_confirm(self):
        self.theta = int(self.angle_slider.value())
        self.close()


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.load_image('res\\niceimage.jpg')
        self.root_layout = QHBoxLayout()

        self.control_view = self.build_control_view()
        self.view_layout = self.build_view_layout()

        self.root_layout.addLayout(self.view_layout, stretch=65)
        self.root_layout.addLayout(self.control_view, stretch=35)

        self.setWindowTitle('ImgProc')
        self.setLayout(self.root_layout)
        self.show()

    @staticmethod
    def build_labeled_slider(text, min, max, init_value, c_slider=None):
        h_layout = QHBoxLayout()

        if c_slider is not None:
            slider = c_slider
        else:
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
                Window.build_labeled_slider('H: ', 0, 360, 0)

        s2, self.saturation_slider_label, self.saturation_slider = \
                Window.build_labeled_slider('S: ', 0, 100, 50)

        s3, self.value_slider_label, self.value_slider = \
                Window.build_labeled_slider('V: ', 0, 100, 50)

        s4, self.sigma_slider_label, self.sigma_slider = Window.build_labeled_slider(
            'SIGMA: ', 0, 20, 0)


        self.hue_slider.valueChanged[int].connect(self.on_image_hsv_change)
        self.saturation_slider.valueChanged[int].connect(self.on_image_hsv_change)
        self.value_slider.valueChanged[int].connect(self.on_image_hsv_change)
        self.sigma_slider.valueChanged[int].connect(self.on_gaus_sigma_change)

        colors_info_layout = QVBoxLayout()
        self.rgb_label = QLabel(ColorsFormats.RGB.format(0, 0, 0))
        self.hsv_label = QLabel(ColorsFormats.HSV.format(0, 0, 0))
        self.lab_label = QLabel(ColorsFormats.LAB.format(0, 0, 0))
        self.l_hist_button = QPushButton('Show hist')
        self.l_hist_button.clicked.connect(self.show_hist)

        colors_info_layout.addWidget(self.rgb_label)
        colors_info_layout.addWidget(self.hsv_label)
        colors_info_layout.addWidget(self.lab_label)

        v_layout.addLayout(s1)
        v_layout.addLayout(s2)
        v_layout.addLayout(s3)
        v_layout.addLayout(s4)
        v_layout.addLayout(colors_info_layout)
        v_layout.addWidget(self.l_hist_button)

        self.operations_list_widget = QListWidget()
        self.operations_list_widget.itemClicked.connect(self.pipeline_item_clicked)
        self.operations_list_widget.setDragEnabled(True)

        l1 = QListWidgetItem('Sobel')
        l1.setCheckState(Qt.Checked)

        l2 = QListWidgetItem('Otsu')
        l2.setCheckState(Qt.Checked)


        l3 = QListWidgetItem('Gabor')
        l3.setData(Qt.UserRole, {'theta': 0})
        l3.setCheckState(Qt.Checked)

        
        
        self.operations_list_widget.insertItem(1, l1)
        self.operations_list_widget.insertItem(2, l2)
        self.operations_list_widget.insertItem(3, l3)

        
        

        h_layout.addLayout(v_layout, stretch=50)
        h_layout.addWidget(self.operations_list_widget, stretch=50)

        h_pipeline = QHBoxLayout()
        buttons_layout = QVBoxLayout()
        buttons_layout.setAlignment(Qt.AlignTop)
        
        self.run_button = QPushButton('run')
        self.run_button.clicked.connect(self.run_pipeline)
        self.open_image_button = QPushButton('open')
        self.open_image_button.clicked.connect(self.open_image)
        self.clear_pipeline_button = QPushButton('clear')
        self.reset_image_button = QPushButton('reset')
        self.reset_image_button.clicked.connect(self.reset_image)


        self.clear_pipeline_button.clicked.connect(self.clear_pipeline)

        buttons_layout.addWidget(self.run_button, stretch=5)
        buttons_layout.addWidget(self.clear_pipeline_button, stretch=5)
        buttons_layout.addWidget(self.reset_image_button, stretch=5)
        buttons_layout.addWidget(self.open_image_button, stretch=5)


        self.pipeline_operations_widget = QListWidget()
        self.pipeline_operations_widget.setAcceptDrops(True)
        self.pipeline_operations_widget.setDragEnabled(True)

        h_pipeline.addWidget(self.pipeline_operations_widget)
        h_pipeline.addLayout(buttons_layout)

        control_view.addLayout(h_layout, stretch=30)
        control_view.addLayout(h_pipeline, stretch=70)

        return control_view

    def reset_image(self):
        self.hsv_adjust_image = self.image
        self.show_image(self.image)

    def clear_pipeline(self):
        self.pipeline_operations_widget.clear()
    
    def run_pipeline(self):
        for index in range(self.pipeline_operations_widget.count()):
            item = self.pipeline_operations_widget.item(index)
            if item.checkState() == Qt.Checked:
                if item.text() == 'Sobel':
                    self.show_image(T.sobel_filter(self.shown_image).numpy())
                elif item.text() == 'Otsu':
                    self.show_image(T.otsu_binarization(self.shown_image).numpy())
                elif item.text() == 'Gabor':
                    self.show_image(T.gabor_filter(self.shown_image,
                            item.data(Qt.UserRole)['theta']).numpy())


    def pipeline_item_clicked(self, item):
        
        if item.text() == 'Gabor':
            gaborDialog = GaborDialog(item.data(Qt.UserRole)['theta'])
            gaborDialog.exec_()
            item.setData(Qt.UserRole, {'theta': gaborDialog.theta})


            

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

    
    def open_image(self):
        path, f = QFileDialog.getOpenFileName(self,
                         'Select image', filter="Images (*.png *.jpg);")
        print(f, path)
        self.load_image(path)
        self.show_image(self.image)

    # TODO check for image channels (must be 3)
    def load_image(self, path):
        self.image = np.array(Image.open(path))
        self.hsv_adjust_image = self.image
        if self.image.shape[-1] == 4:
            self.image = self.image[:,:,:3]

    
    def show_hist(self):

        l_comp = T.cielab_L_component(self.shown_image).numpy()

        self.hist_figure.clear()
        axs = self.hist_figure.add_subplot(111)
        axs.hist(l_comp.flatten(), bins=100)
        self.hist_canvas.draw()

    
    def mouse_moved_on_image(self, mouse_event):
        if not mouse_event.inaxes:
            return
        
        x, y = mouse_event.xdata, mouse_event.ydata

        r, g, b = map(int, self.shown_image[int(y), int(x)])
        h, s, v = T.rgb_to_hsv((r, g, b))
        L, la, lb = T.rgb_to_cielab((r, g, b))

        self.rgb_label.setText(ColorsFormats.RGB.format(r, g, b))
        self.hsv_label.setText(ColorsFormats.HSV.format(h, s, v))
        self.lab_label.setText(ColorsFormats.LAB.format(L, la, lb))

    def on_gaus_sigma_change(self, value):
        val = self.sigma_slider.value()

        val = (val / self.sigma_slider.maximum()) * 10

        self.sigma_slider_label.setText(str(val))

        if val == 0.0:
            self.show_image(self.hsv_adjust_image)
            return

        self.show_image(T.gaussian_filter(self.hsv_adjust_image, val, 5).numpy())


    def on_image_hsv_change(self, value):
        h = self.hue_slider.value()
        s = self.saturation_slider.value()
        v = self.value_slider.value()

        self.hue_slider_label.setText(str(h)) 
        self.saturation_slider_label.setText(str(s))
        self.value_slider_label.setText(str(v))

        s = s / 100.0 + 0.5
        v = v / 100.0 + 0.5
        self.hsv_adjust_image = T.transform_hsv(self.image, h, s, v).numpy()
        self.show_image(self.hsv_adjust_image)

    def show_image(self, image):
        self.image_figure.clear()
        axs = self.image_figure.add_subplot(111)
        axs.imshow(image, interpolation='none')
        self.shown_image = image
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
