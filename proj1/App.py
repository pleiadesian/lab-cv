import tkinter
import tkinter.filedialog
import tkinter.ttk as ttk
import cv2
import numpy as np
from util import convolution, filter
from PIL import Image, ImageTk


class ImageProc():
    def __init__(self):
        self.image = None

    def choose_image(self):
        select_filename = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件

        self.image = cv2.imread(select_filename, cv2.IMREAD_GRAYSCALE)
        image_roberts = convolution.robert_edge(self.image)
        image_sobel_x = convolution.sobel_edge_x(self.image)
        image_sobel_y = convolution.sobel_edge_y(self.image)
        image_prewitt = convolution.prewitt_edge(self.image)
        if not kernel.get().isdigit():
            image_mean = filter.mean_filter(self.image)
            image_median = filter.median_filter(self.image)
        else:
            k = int(kernel.get())
            image_mean = filter.mean_filter(self.image, np.ones((k, k)))
            image_median = filter.median_filter(self.image, np.ones((k, k)))
        if not sigma.get().isdigit():
            image_gaussian = filter.mean_filter(self.image, filter.gaussian_mat(1))
        else:
            image_gaussian = filter.mean_filter(self.image, filter.gaussian_mat(int(sigma.get())))
        self.showImg(self.image, img_original)
        self.showImg(image_roberts, img_roberts)
        self.showImg(image_sobel_x, img_sobel_x)
        self.showImg(image_sobel_y, img_sobel_y)
        self.showImg(image_prewitt, img_prewitt)
        self.showImg(image_mean, img_mean)
        self.showImg(image_median, img_median)
        self.showImg(image_gaussian, img_gaussian)

    def redo_gaussian(self):
        if not sigma.get().isdigit():
            image_gaussian = filter.mean_filter(self.image, filter.gaussian_mat(1))
        else:
            image_gaussian = filter.mean_filter(self.image, filter.gaussian_mat(int(sigma.get())))
        self.showImg(image_gaussian, img_gaussian)

    def redo_filter(self):
        if not kernel.get().isdigit():
            image_mean = filter.mean_filter(self.image)
            image_median = filter.median_filter(self.image)
        else:
            k = int(kernel.get())
            image_mean = filter.mean_filter(self.image, np.ones((k, k)))
            image_median = filter.median_filter(self.image, np.ones((k, k)))
        self.showImg(image_mean, img_mean)
        self.showImg(image_median, img_median)

    def showImg(self, img, outputimg):
        img_temp = Image.fromarray(img)
        render = ImageTk.PhotoImage(image=img_temp)
        outputimg.config(image=render)
        outputimg.image = render


if __name__ == '__main__':
    ip = ImageProc()

    top = tkinter.Tk()
    top.title = 'new'
    top.geometry('1900x800')

    style = ttk.Style()
    style.configure("TButton", foreground="blue", background="orange")

    img_original = tkinter.Label(top)
    img_roberts = tkinter.Label(top)
    img_sobel_x = tkinter.Label(top)
    img_sobel_y = tkinter.Label(top)
    img_prewitt = tkinter.Label(top)
    img_mean = tkinter.Label(top)
    img_median = tkinter.Label(top)
    img_gaussian = tkinter.Label(top)

    left_padding = 5
    gap = 420
    img_original.place(x=left_padding, y=80)
    img_roberts.place(x=left_padding, y=450)
    img_sobel_x.place(x=left_padding+gap, y=80)
    img_sobel_y.place(x=left_padding+gap, y=450)
    img_prewitt.place(x=left_padding+2*gap, y=80)
    img_mean.place(x=left_padding+2*gap, y=450)
    img_median.place(x=left_padding+3*gap, y=80)
    img_gaussian.place(x=left_padding+3*gap, y=450)

    left_padding_l = left_padding + 20
    label_origin = tkinter.Label(top, text="original", fg='black')
    label_roberts = tkinter.Label(top, text="roberts operation", fg='black')
    label_sobel_x = tkinter.Label(top, text="sobel operation (x)", fg='black')
    label_sobel_y = tkinter.Label(top, text="sobel operation (y)", fg='black')
    label_prewitt = tkinter.Label(top, text="prewitt operation", fg='black')
    label_mean = tkinter.Label(top, text="mean filter", fg='black')
    label_median = tkinter.Label(top, text="median filter", fg='black')
    label_gaussian = tkinter.Label(top, text="gaussian filter", fg='black')

    label_origin.place(x=left_padding_l, y=50)
    label_roberts.place(x=left_padding_l, y=420)
    label_sobel_x.place(x=left_padding_l+gap, y=50)
    label_sobel_y.place(x=left_padding_l+gap, y=420)
    label_prewitt.place(x=left_padding_l+2*gap, y=50)
    label_mean.place(x=left_padding_l+2*gap, y=420)
    label_median.place(x=left_padding_l+3*gap, y=50)
    label_gaussian.place(x=left_padding_l+3*gap, y=420)

    label_sigma = tkinter.Label(top, text="enter sigma for Gaussian filter (default 1):", fg='black')
    label_sigma.place(x=256, y=5)
    sigma = tkinter.StringVar()
    sigma_entry = tkinter.Entry(top, width=10, textvariable=sigma)
    sigma_entry.place(x=550, y=0)
    sigma_button = ttk.Button(top, text="redo Gaussian filter", command=ip.redo_gaussian, style='TButton')
    sigma_button.place(x=700, y=5)
    label_sigma = tkinter.Label(top, text="enter kernel size for all filters (default 3x3):", fg='black')
    label_sigma.place(x=250, y=32)
    kernel = tkinter.StringVar()
    kernel_entry = tkinter.Entry(top, width=10, textvariable=kernel)
    kernel_entry.place(x=550, y=27)
    kernel_button = ttk.Button(top, text="redo all filters", command=ip.redo_filter, style='TButton')
    kernel_button.place(x=700, y=27)

    submit_button = ttk.Button(top, text="choose image", command=ip.choose_image, style='TButton')
    submit_button.place(x=10, y=5)

    top.mainloop()
