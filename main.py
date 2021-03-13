import matplotlib.pyplot as plt
from selection_widgets import RectangleSelector, LineSelector
import numpy as np
import cv2
from skimage import draw, color, io, morphology
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

def ChooseRectangle(image, fname):
    rect = None
    def onselect(eclick, erelease):
        nonlocal rect
        rect = (
            int(min(eclick.xdata, erelease.xdata)),
            int(max(eclick.xdata, erelease.xdata)),
            int(min(eclick.ydata, erelease.ydata)),
            int(max(eclick.ydata, erelease.ydata)),
        )
        plt.close()

    plt.imshow(image)
    rs = RectangleSelector(plt.gca(), onselect)
    plt.title('Select Rectangle')
    plt.suptitle(fname)
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    return rect

def DrawSegmentation(image, fname):
    mask = np.full(image.shape[:2], cv2.GC_PR_BGD, np.uint8)

    def draw_mask():
        seg = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
        img = image.copy()
        if np.any(seg):
            img[seg, 0] = 255
            img[np.logical_not(seg), 2] = 255
        obj.set_data(img)
        plt.draw()
    def onselect(geometry, button):
        nonlocal mask
        initial_point, end_point = geometry
        rr, cc = draw.line(*initial_point, *end_point)
        flag = cv2.GC_FGD if button == 1 else cv2.GC_BGD
        mask[cc, rr] = flag
        bgModel = np.zeros((1,65), np.float64)
        fgModel = np.zeros((1,65), np.float64)
        mask, _, _ = cv2.grabCut(image, mask, None, bgModel, fgModel, 5, cv2.GC_INIT_WITH_MASK)
        draw_mask()
    def key_pressed(ev):
        if ev.key in ('+', '-'):
            flag1 = cv2.GC_FGD if ev.key == '+' else cv2.GC_BGD
            flag2 = cv2.GC_PR_FGD if ev.key == '+' else cv2.GC_PR_BGD
            seg = np.logical_or(mask == flag1, mask == flag2)
            seg = morphology.binary_dilation(seg)
            mask[seg] = flag1
            draw_mask()
        if ev.key in ('enter', 'escape'):
            plt.close()

    obj = plt.imshow(image)
    rs1 = LineSelector(plt.gca(), lambda g: onselect(g, 1), 'line', button=1, maxdist=5, line_props={'color': 'red', 'alpha': 1})
    rs2 = LineSelector(plt.gca(), lambda g: onselect(g, 3), 'line', button=3, maxdist=5, line_props={'color': 'blue', 'alpha': 1})
    plt.gcf().canvas.mpl_connect('key_press_event', key_pressed)
    plt.title('Foreground (left-click), Background (right-click), +/- (dilation/erosion)')
    plt.suptitle(fname)
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    return np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)

Tk().withdraw()
filenames = askopenfilenames()

for i, fname in enumerate(filenames):
    image = color.gray2rgb(io.imread(fname, True))
    title = f'{fname} {i+1}/{len(filenames)}'
    rect = ChooseRectangle(image, title)
    seg = DrawSegmentation(image[rect[2]:rect[3], rect[0]:rect[1]], title)

    mask = np.zeros(image.shape[:2], np.uint8)
    mask[rect[2]:rect[3], rect[0]:rect[1]] = seg*255
    io.imsave(fname[:-4] + '-seg.png', mask, check_contrast=False)
