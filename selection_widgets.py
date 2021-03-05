# https://gist.github.com/blink1073/6ecb48889d3c7526f3c5

import numpy as np
from matplotlib.widgets import AxesWidget
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.transforms import blended_transform_factory
LABELS_CMAP = mcolors.ListedColormap(['white', 'red', 'dodgerblue', 'gold',
                                      'greenyellow', 'blueviolet'])


class SelectionWidget(AxesWidget):

    """Base class for selection widgets"""

    def __init__(self, ax, onselect,  useblit=True, button=None):
        AxesWidget.__init__(self, ax)

        self.visible = True
        self.connect_event('motion_notify_event', self._onmove)
        self.connect_event('button_press_event', self._press)
        self.connect_event('button_release_event', self._release)
        self.connect_event('draw_event', self.update_background)
        self.connect_event('key_press_event', self._on_key_press)
        self.connect_event('scroll_event', self._on_scroll)

        # for activation / deactivation
        self.active = True
        self.background = None
        self.artists = []

        self.onselect = onselect
        self.useblit = useblit and self.canvas.supports_blit

        if button is None or isinstance(button, list):
            self.validButtons = button
        elif isinstance(button, int):
            self.validButtons = [button]

        # will save the data (position at mouseclick)
        self.eventpress = None
        # will save the data (pos. at mouserelease)
        self.eventrelease = None

    def update_background(self, event):
        """force an update of the background"""
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def ignore(self, event):
        """return *True* if *event* should be ignored"""
        if not self.active:
            return True

        # If canvas was locked
        if not self.canvas.widgetlock.available(self):
            return True

        if not hasattr(event, 'button'):
            event.button = None

        # Only do rectangle selection if event was triggered
        # with a desired button
        if self.validButtons is not None:
            if not event.button in self.validButtons:
                return True

        # If no button was pressed yet ignore the event if it was out
        # of the axes
        if self.eventpress is None:
            return event.inaxes != self.ax

        # If a button was pressed, check if the release-button is the
        # same. If event is out of axis, limit the data coordinates to axes
        # boundaries.
        if event.button == self.eventpress.button and event.inaxes != self.ax:
            (xdata, ydata) = self.ax.transData.inverted().transform_point(
                (event.x, event.y))
            x0, x1 = self.ax.get_xbound()
            y0, y1 = self.ax.get_ybound()
            xdata = max(x0, xdata)
            xdata = min(x1, xdata)
            ydata = max(y0, ydata)
            ydata = min(y1, ydata)
            event.xdata = xdata
            event.ydata = ydata
            return False

        # If a button was pressed, check if the release-button is the
        # same.
        return (event.inaxes != self.ax or
                event.button != self.eventpress.button)

    def update(self):
        """draw using newfangled blit or oldfangled draw depending on
        useblit

        """
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            for artist in self.artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False

    def _press(self, event):
        """Button press event"""
        if not self.ignore(event):
            self.eventpress = event
            self.press(event)

    def press(self, event):
        """Button press handler"""
        pass

    def _release(self, event):
        if not self.ignore(event) and not self.eventpress is None:
            self.eventrelease = event
            # TODO: handle minimum size - add a clear method
            self.release(event)
            self.eventpress = None
            self.eventrelease = None

    def release(self, event):
        """Button release event"""
        pass

    def _onmove(self, event):
        if not self.ignore(event):
            self.onmove(event)

    def onmove(self, event):
        """Cursor motion event"""
        pass

    def _on_scroll(self, event):
        if not self.ignore(event):
            self.on_scroll(event)

    def on_scroll(self, event):
        """Mouse scroll event"""
        pass

    def _on_key_press(self, event):
        if not self.ignore(event):
            self.on_key_press(event)

    def on_key_press(self, event):
        """Key press event"""
        pass

    def set_active(self, active):
        """
        Use this to activate / deactivate the Selector
        from your program with an boolean parameter *active*.
        """
        self.active = active

    def get_active(self):
        """ Get status of active mode (boolean variable)"""
        return self.active

    def set_visible(self, visible):
        """ Set the visibility of our artists """
        for artist in self.artists:
            artist.set_visible(visible)

    def draw_rubberband(self, x0, x1, y0, y1):
        """Draw a box using the native toolkit given data coordinates
        """
        height = self.canvas.figure.bbox.height

        x0, y0 = self.ax.transData.transform([x0, y0])
        x1, y1 = self.ax.transData.transform([x1, y1])

        y1 = height - y1
        y0 = height - y0

        w = abs(x1 - x0)
        h = abs(y1 - y0)

        rect = [int(val)for val in (min(x0, x1), min(y0, y1), w, h)]
        self.canvas.drawRectangle(rect)


class ToolHandles(object):

    """Control handles for canvas tools.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes where tool handles are displayed.
    x, y : 1D arrays
        Coordinates of control handles.
    marker : str
        Shape of marker used to display handle. See `matplotlib.pyplot.plot`.
    marker_props : dict
        Additional marker properties. See :class:`matplotlib.lines.Line2D`.
    """

    def __init__(self, ax, x, y, marker='o', marker_props=None, useblit=True):
        self.ax = ax

        props = dict(marker=marker, markersize=7, mfc='w', ls='none',
                     alpha=0.5, visible=False)
        props.update(marker_props if marker_props is not None else {})
        self._markers = Line2D(x, y, animated=useblit, **props)
        self.ax.add_line(self._markers)
        self.artist = self._markers

    @property
    def x(self):
        return self._markers.get_xdata()

    @property
    def y(self):
        return self._markers.get_ydata()

    def set_data(self, pts, y=None):
        """Set x and y positions of handles"""
        if y is not None:
            x = pts
            pts = np.array([x, y])
        self._markers.set_data(pts)

    def set_visible(self, val):
        self._markers.set_visible(val)

    def set_animated(self, val):
        self._markers.set_animated(val)

    def closest(self, x, y):
        """Return index and pixel distance to closest index."""
        pts = np.transpose((self.x, self.y))
        # Transform data coordinates to pixel coordinates.
        pts = self.ax.transData.transform(pts)
        diff = pts - ((x, y))
        if diff.ndim == 2:
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            return np.argmin(dist), np.min(dist)
        else:
            return 0, np.sqrt(np.sum(diff ** 2))


class RectangleSelector(SelectionWidget):

    _shape_klass = Rectangle

    def __init__(self, ax, onselect, drawtype='patch',
                 minspanx=None, minspany=None, useblit=True,
                 lineprops=None, rectprops=None, spancoords='data',
                 button=1, maxdist=10, marker_props=None):
        SelectionWidget.__init__(self, ax, onselect=onselect, useblit=useblit, 
                                                          button=button)

        self.to_draw = None
        self.visible = True

        if drawtype == 'box':  # backwards compatibility
            drawtype = 'patch'

        if drawtype == 'none':
            drawtype = 'line'                        # draw a line but make it
            self.visible = False                     # invisible

        if drawtype == 'patch':
            if rectprops is None:
                rectprops = dict(facecolor='white', edgecolor='black',
                                 alpha=0.5, fill=False)
            self.rectprops = rectprops
            self.to_draw = self._shape_klass((0, 0),
                                     0, 1, visible=False, **self.rectprops)
            self.ax.add_patch(self.to_draw)
        if drawtype == 'line':
            if lineprops is None:
                lineprops = dict(color='black', linestyle='-',
                                 linewidth=2, alpha=0.5)
            self.lineprops = lineprops
            self.to_draw = Line2D([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], visible=False,
                                  **self.lineprops)
            self.ax.add_line(self.to_draw)

        self.minspanx = minspanx
        self.minspany = minspany

        assert(spancoords in ('data', 'pixels'))

        self.spancoords = spancoords
        self.drawtype = drawtype
        self.maxdist = maxdist

        if rectprops is None:
            props = dict(mec='r')
        else:
            props = dict(mec=rectprops['edgecolor'])
        self._corner_order = ['NW', 'NE', 'SE', 'SW']
        xc, yc = self.corners
        self._corner_handles = ToolHandles(self.ax, xc, yc, marker_props=props,
                                           useblit=self.useblit)

        self._edge_order = ['W', 'N', 'E', 'S']
        xe, ye = self.edge_centers
        self._edge_handles = ToolHandles(self.ax, xe, ye, marker='s',
                                         marker_props=props, useblit=self.useblit)

        xc, yc = self.center
        self._center_handle = ToolHandles(self.ax, [xc], [yc], marker='s',
                                          marker_props=props, useblit=self.useblit)

        self.artists = [self.to_draw, self._center_handle.artist,
                        self._corner_handles.artist,
                        self._edge_handles.artist]

    @property
    def _rect_bbox(self):
        if self.drawtype == 'patch':
            x0 = self.to_draw.get_x()
            y0 = self.to_draw.get_y()
            width = self.to_draw.get_width()
            height = self.to_draw.get_height()
            return x0, y0, width, height
        else:
            x, y = self.to_draw.get_data()
            x0, x1 = min(x), max(x)
            y0, y1 = min(y), max(y)
            return x0, y0, x1 - x0, y1 - y0

    @property
    def corners(self):
        """Corners of rectangle from lower left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox
        xc = x0, x0 + width, x0 + width, x0
        yc = y0, y0, y0 + height, y0 + height
        return xc, yc

    @property
    def edge_centers(self):
        """Midpoint of rectangle edges from left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox
        w = width / 2.
        h = height / 2.
        xe = x0, x0 + w, x0 + width, x0 + w
        ye = y0 + h, y0, y0 + h, y0 + height
        return xe, ye

    @property
    def center(self):
        """Center of rectangle"""
        x0, y0, width, height = self._rect_bbox
        return x0 + width / 2., y0 + height / 2.

    @property
    def extents(self):
        """Return (xmin, xmax, ymin, ymax)."""
        x0, y0, width, height = self._rect_bbox
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    @extents.setter
    def extents(self, extents):
        # Update displayed shape
        self.draw_shape(extents)
        # Update displayed handles
        self._corner_handles.set_data(*self.corners)
        self._edge_handles.set_data(*self.edge_centers)
        self._center_handle.set_data(*self.center)

        self.set_visible(self.visible)

        if self.eventpress:
            self.draw_rubberband(*extents)

    def draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])

        if self.drawtype == 'patch':
            self.to_draw.set_x(xmin)
            self.to_draw.set_y(ymin)
            self.to_draw.set_width(xmax - xmin)
            self.to_draw.set_height(ymax - ymin)

        elif self.drawtype == 'line':
            self.to_draw.set_data([xmin,  xmin, xmax, xmax, xmin],
                                  [ymin, ymax, ymax, ymin, ymin])

    def release(self, event):
        self._extents_on_press = None

        # release coordinates, button, ...
        self.eventrelease = event

        if self.spancoords == 'data':
            xmin, ymin = self.eventpress.xdata, self.eventpress.ydata
            xmax, ymax = self.eventrelease.xdata, self.eventrelease.ydata
            # calculate dimensions of box or line get values in the right
            # order
        elif self.spancoords == 'pixels':
            xmin, ymin = self.eventpress.x, self.eventpress.y
            xmax, ymax = self.eventrelease.x, self.eventrelease.y
        else:
            raise ValueError('spancoords must be "data" or "pixels"')

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        spanx = xmax - xmin
        spany = ymax - ymin
        xproblems = self.minspanx is not None and spanx < self.minspanx
        yproblems = self.minspany is not None and spany < self.minspany

        if (self.drawtype in ['patch', 'line'] and (xproblems or yproblems)):
            # check if drawn distance (if it exists) is not too small in
            # neither x nor y-direction
            return

        # update the eventpress and eventrelease with the resulting extents
        x1, x2, y1, y2 = self.extents
        self.eventpress.xdata = x1
        self.eventpress.ydata = y1
        xy1 = self.ax.transData.transform_point([x1, y1])
        self.eventpress.x, self.eventpress.y = xy1

        self.eventrelease.xdata = x2
        self.eventrelease.ydata = y2
        xy2 = self.ax.transData.transform_point([x2, y2])
        self.eventrelease.x, self.eventrelease.y = xy2

        self.onselect(self.eventpress, self.eventrelease)
                                              # call desired function
        self.update()
        return False

    def press(self, event):
        """on button press event"""
        # make the drawed box/line visible get the click-coordinates,
        # button, ...
        self.set_visible(self.visible)
        self._set_active_handle(event)
        if self.active_handle is None:
            # Clear previous rectangle before drawing new rectangle.
            self.set_visible(False)
            self.update()
        self.set_visible(self.visible)

    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event"""
        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        c_idx, c_dist = self._corner_handles.closest(event.x, event.y)
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        m_idx, m_dist = self._center_handle.closest(event.x, event.y)

        if event.key in ['alt', ' ']:
            self.active_handle = 'C'
            self._extents_on_press = self.extents

        # Set active handle as closest handle, if mouse click is close enough.
        elif m_dist < self.maxdist:
            self.active_handle = 'C'
        elif c_dist > self.maxdist and e_dist > self.maxdist:
            self.active_handle = None
            return
        elif c_dist < e_dist:
            self.active_handle = self._corner_order[c_idx]
        else:
            self.active_handle = self._edge_order[e_idx]

        # Save coordinates of rectangle at the start of handle movement.
        x1, x2, y1, y2 = self.extents
        # Switch variables so that only x2 and/or y2 are updated on move.
        if self.active_handle in ['W', 'SW', 'NW']:
            x1, x2 = x2, event.xdata
        if self.active_handle in ['N', 'NW', 'NE']:
            y1, y2 = y2, event.ydata
        self._extents_on_press = x1, x2, y1, y2

    def onmove(self, event):
        if self.eventpress is None:
            return

        key = self.eventpress.key or ''

        # resize an existing shape
        if self.active_handle and not self.active_handle == 'C':
            x1, x2, y1, y2 = self._extents_on_press
            if self.active_handle in ['E', 'W'] + self._corner_order:
                x2 = event.xdata
            if self.active_handle in ['N', 'S'] + self._corner_order:
                y2 = event.ydata

        # move existing shape
        elif self.active_handle == 'C':
            x1, x2, y1, y2 = self._extents_on_press
            dx = event.xdata - self.eventpress.xdata
            dy = event.ydata - self.eventpress.ydata
            x1 += dx
            x2 += dx
            y1 += dy
            y2 += dy

        # new shape
        else:
            center = [self.eventpress.xdata, self.eventpress.ydata]
            center_pix = [self.eventpress.x, self.eventpress.y]
            dx = (event.xdata - center[0]) / 2.
            dy = (event.ydata - center[1]) / 2.

            # square shape
            if 'shift' in key:
                dx_pix = abs(event.x - center_pix[0])
                dy_pix = abs(event.y - center_pix[1])
                if not dx_pix:
                    return
                maxd = max(abs(dx_pix), abs(dy_pix))
                if abs(dx_pix) < maxd:
                    dx *= maxd / abs(dx_pix)
                if abs(dy_pix) < maxd:
                    dy *= maxd / abs(dy_pix)

            # from center
            if key == 'control' or key == 'ctrl+shift':
                dx *= 2
                dy *= 2

            # from corner
            else:
                center[0] += dx
                center[1] += dy

            x1, x2, y1, y2 = (center[0] - dx, center[0] + dx,
                              center[1] - dy, center[1] + dy)

        self.extents = x1, x2, y1, y2


class EllipseSelector(RectangleSelector):

    _shape_klass = Ellipse

    def draw_shape(self, extents):
        x1, x2, y1, y2 = extents
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        center = [x1 + (x2 - x1) / 2., y1 + (y2 - y1) / 2.]
        a = (xmax - xmin) / 2.
        b = (ymax - ymin) / 2.

        if self.drawtype == 'patch':
            self.to_draw.center = center
            self.to_draw.width = 2 * a
            self.to_draw.height = 2 * b
        else:
            rad = np.arange(31) * 12 * np.pi / 180
            x = a * np.cos(rad) + center[0]
            y = b * np.sin(rad) + center[1]
            self.to_draw.set_data(x, y)

    @property
    def _rect_bbox(self):
        if self.drawtype == 'patch':
            x, y = self.to_draw.center
            width = self.to_draw.width
            height = self.to_draw.height
            return x - width / 2., y - height / 2., width, height
        else:
            x, y = self.to_draw.get_data()
            x0, x1 = min(x), max(x)
            y0, y1 = min(y), max(y)
            return x0, y0, x1 - x0, y1 - y0

    @property
    def geometry(self):
        x0, y0, width, height = self._rect_bbox
        return x0 + width / 2., y0 + width / 2., width, height


class LassoSelector(SelectionWidget):
    """Selection curve of an arbitrary shape.
    """
    def __init__(self, ax, onselect, useblit=True, button=None, 
        lineprops=None):
        SelectionWidget.__init__(self, ax, onselect=onselect, useblit=useblit, 
            button=button)

        self.verts = None

        if lineprops is None:
            lineprops = dict()
        self.line = Line2D([], [], **lineprops)
        self.line.set_visible(False)
        self.ax.add_line(self.line)
        self.artists = [self.line]

    def press(self, event):
        if not event.key == 'shift' or self.verts is None:
            self.verts = [(event.xdata, event.ydata)]

        self.line.set_visible(True)

    def finish(self, event):
            self.verts.append(self.verts[0])
            self.line.set_data(list(zip(*self.verts)))
            self.update()
            self.onselect(self.verts)
            self.verts = None

    def release(self, event):
        self.verts.append((event.xdata, event.ydata))
        if event.key != 'shift':
            self.finish(event)
        else:
            self.verts.append((event.xdata, event.ydata))
            self.line.set_data(list(zip(*self.verts)))
            self.update()

    def onmove(self, event):
        if self.verts is None:
            return
        if event.key == 'shift':
            self.verts[-1] = [event.xdata, event.ydata]
        elif event.button:
            self.verts.append((event.xdata, event.ydata))
        else:
            return self.finish(event)
        self.line.set_data(list(zip(*self.verts)))
        self.update()


class LineSelector(SelectionWidget):

    def __init__(self, ax, onselect, useblit=True, button=1,
                           maxdist=10, line_props=None):

        super(LineSelector, self).__init__(ax, onselect,
            useblit=useblit,  button=button)

        props = dict(color='r', linewidth=1, alpha=0.4, solid_capstyle='butt')
        props.update(line_props if line_props is not None else {})
        self.linewidth = props['linewidth']
        self.maxdist = maxdist
        self._active_pt = None

        x = (0, 0)
        y = (0, 0)
        self._end_pts = np.transpose([x, y])

        self._line = Line2D(x, y, visible=False, animated=True, **props)
        self.ax.add_line(self._line)

        self._handles = ToolHandles(self.ax, x, y, useblit=useblit)
        self._handles.set_visible(False)
        self.artists = [self._line, self._handles.artist]

    @property
    def end_points(self):
        return self._end_pts.astype(int)

    @end_points.setter
    def end_points(self, pts):

        self._end_pts = pts = np.asarray(pts)
        self._line.set_data(np.transpose(pts))
        self._line.set_linewidth(self.linewidth)

        self._center = center = (pts[1] + pts[0]) / 2.
        handle_pts = np.vstack((pts[0], center, pts[1])).T
        self._handles.set_data(handle_pts)

        self.set_visible(True)
        self.update()

    def press(self, event):
        idx, px_dist = self._handles.closest(event.x, event.y)
        if px_dist < self.maxdist:
            self._active_pt = idx
        else:
            self._active_pt = None

        if event.key in ['alt', ' ']:
            self._active_pt = 1

        self.set_visible(True)

        if self._active_pt is None:
            self._active_pt = 0
            x, y = event.xdata, event.ydata
            self._end_pts = np.array([[x, y], [x, y]])

    def release(self, event):
        self._active_pt = None
        self.onselect(self.geometry)

    def onmove(self, event):
        if self._active_pt is None:
            return
        self.update_data(event.xdata, event.ydata)

    def update_data(self, x=None, y=None):
        if x is not None:
            # check for center
            if self._active_pt == 1:
                xc, yc = self._center
                xo, yo = x - xc, y - yc
                self._end_pts += [xo, yo]
            elif self._active_pt == 0:
                self._end_pts[0, :] = x, y
            else:
                self._end_pts[1, :] = x, y
        self.end_points = self._end_pts

    @property
    def geometry(self):
        return self.end_points

    def on_scroll(self, event):
        if event.button == 'up':
            self._thicken_scan_line()
        elif event.button == 'down':
            self._shrink_scan_line()

    def on_key_press(self, event):
        if event.key == '+':
            self._thicken_scan_line()
        elif event.key == '-':
            self._shrink_scan_line()

    def _thicken_scan_line(self):
        self.linewidth += 1
        self.update_data()

    def _shrink_scan_line(self):
        if self.linewidth > 1:
            self.linewidth -= 1
            self.update_data()


class PaintSelector(SelectionWidget):
    def __init__(self, ax, onselect, overlay_shape, radius=5, alpha=0.3,
                 rect_props=None, useblit=False, button=None):
        super(PaintSelector, self).__init__(ax, onselect,
            useblit=useblit,  button=button)

        props = dict(edgecolor='r', facecolor='0.7', alpha=0.5,
            animated=useblit)
        props.update(rect_props if rect_props is not None else {})

        self.alpha = alpha
        self.cmap = LABELS_CMAP
        self._overlay_plot = None
        self._radius = radius

        self._cursor = [0, 0, 0]

        # These can only be set after initializing `_cursor`
        self.shape = overlay_shape
        self.label = 1
        self.radius = radius

        # Note that the order is important: Redraw cursor *after* overlay
        self.artists = [self._overlay_plot]

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        if value >= self.cmap.N:
            raise ValueError('Maximum label value = %s' % len(self.cmap - 1))
        self._label = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        self._radius = r
        self._width = 2 * r + 1
        [x, y, _] = self._cursor
        self._cursor = [x, y, self._width]
        self.window = CenteredWindow(r, self._shape)
        self.update()

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, image):
        self._overlay = image
        if image is None:
            self.ax.images.remove(self._overlay_plot)
            self._overlay_plot = None
        elif self._overlay_plot is None:
            props = dict(cmap=self.cmap, alpha=self.alpha,
                         norm=mcolors.NoNorm(), animated=True)
            self._overlay_plot = self.ax.imshow(image, **props)
        else:
            self._overlay_plot.set_data(image)
        self._shape = image.shape
        # this triggers an update
        self.radius = self._radius

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        if not self._overlay_plot is None:
            self._overlay_plot.set_extent((-0.5, shape[1] + 0.5,
                                           shape[0] + 0.5, -0.5))
            self.radius = self._radius
        self.overlay = np.zeros(shape, dtype='uint8')

    def press(self, event):
        self.update_cursor(event.xdata, event.ydata)
        self.update_overlay(event.xdata, event.ydata)

    def release(self, event):
        self.onselect(self.geometry)

    def onmove(self, event):
        self.update_cursor(event.xdata, event.ydata)
        if not self.eventpress:
            x, y, r = self._cursor
            self.draw_rubberband(x, x + r, y, y + r)
            return
        self.update_overlay(event.xdata, event.ydata)

    def on_scroll(self, event):
        if event.button == 'up':
            self.radius += 1
        elif event.button == 'down':
            self.radius = max(self.radius - 1, 1)

    def on_key_press(self, event):
        if event.key == '+':
            self.radius += 1
        elif event.key == '-':
            self.radius = max(self.radius - 1, 1)

    def update_overlay(self, x, y):
        overlay = self.overlay
        overlay[self.window.at(y, x)] = self.label
        # Note that overlay calls `update`
        self.overlay = overlay

    def update_cursor(self, x, y):
        x = x - self.radius - 1
        y = y - self.radius - 1
        self._cursor = [x, y, self._width]

    @property
    def geometry(self):
        return self.overlay


class CenteredWindow(object):
    """Window that create slices numpy arrays over 2D windows.

    Examples
    --------
    >>> a = np.arange(16).reshape(4, 4)
    >>> w = CenteredWindow(1, a.shape)
    >>> a[w.at(1, 1)]
    array([[ 0,  1,  2],
           [ 4,  5,  6],
           [ 8,  9, 10]])
    >>> a[w.at(0, 0)]
    array([[0, 1],
           [4, 5]])
    >>> a[w.at(4, 3)]
    array([[14, 15]])
    """
    def __init__(self, radius, array_shape):
        self.radius = radius
        self.array_shape = array_shape

    def at(self, row, col):
        h, w = self.array_shape
        r = self.radius
        xmin = max(0, col - r)
        xmax = min(w, col + r + 1)
        ymin = max(0, row - r)
        ymax = min(h, row + r + 1)
        return [slice(ymin, ymax), slice(xmin, xmax)]


class SpanSelector(SelectionWidget):
    """
    Select a min/max range of the x or y axes for a matplotlib Axes

    Example usage::

        ax = subplot(111)
        ax.plot(x,y)

        def onselect(vmin, vmax):
            print vmin, vmax
        span = SpanSelector(ax, onselect, 'horizontal')

    *onmove_callback* is an optional callback that is called on mouse
      move within the span range

    """

    def __init__(self, ax, onselect, direction, minspan=None, useblit=False,
                 rectprops=None, onmove_callback=None, button=1):
        """
        Create a span selector in *ax*.  When a selection is made, clear
        the span and call *onselect* with::

            onselect(vmin, vmax)

        and clear the span.

        *direction* must be 'horizontal' or 'vertical'

        If *minspan* is not *None*, ignore events smaller than *minspan*

        The span rectangle is drawn with *rectprops*; default::
          rectprops = dict(facecolor='red', alpha=0.5)

        Set the visible attribute to *False* if you want to turn off
        the functionality of the span selector
        """

        '''CHANGES: no more newaxis (what was that for?)
        Do not ignore when invisible - that is what active is for - may still want updates

        TODO: allow the user to hold shift and move the cursor
        '''
        SelectionWidget.__init__(self, ax, onselect, button=button, useblit=useblit)

        if rectprops is None:
            rectprops = dict(facecolor='red', alpha=0.5)

        assert direction in ['horizontal', 'vertical'], 'Must choose horizontal or vertical for direction'
        self.direction = direction

        self.pressv = None

        self.rectprops = rectprops
        self.onmove_callback = onmove_callback
        self.minspan = minspan

        # Needed when dragging out of axes
        self.prev = (0, 0)

        if self.direction == 'horizontal':
            trans = blended_transform_factory(self.ax.transData,
                                              self.ax.transAxes)
            w, h = 0, 1
        else:
            trans = blended_transform_factory(self.ax.transAxes,
                                              self.ax.transData)
            w, h = 1, 0
        self.rect = Rectangle((0, 0), w, h,
                              transform=trans,
                              visible=False,
                              **self.rectprops)

        if not self.useblit:
            self.ax.add_patch(self.rect)

        self.artists = [self.rect]

    def press(self, event):
        """on button press event"""
        self.rect.set_visible(self.visible)
        if self.direction == 'horizontal':
            self.pressv = event.xdata
        else:
            self.pressv = event.ydata
        return False

    def release(self, event):
        """on button release event"""
        self.rect.set_visible(self.visible)
        self.update()
        vmin = self.pressv
        if self.direction == 'horizontal':
            vmax = event.xdata or self.prev[0]
        else:
            vmax = event.ydata or self.prev[1]

        if vmin > vmax:
            vmin, vmax = vmax, vmin
        span = vmax - vmin
        if self.minspan is not None and span < self.minspan:
            return
        self.onselect(vmin, vmax)
        return False

    def onmove(self, event):
        """on motion notify event"""
        if not self.eventpress:
            return

        x, y = event.xdata, event.ydata
        self.prev = x, y
        if self.direction == 'horizontal':
            v = x
        else:
            v = y

        if self.eventpress.key in ['alt', 'shift', ' ']:
            # center the window where the cursor is
            # make sure to handle onmove_callback as well
            return

        minv, maxv = v, self.pressv
        if minv > maxv:
            minv, maxv = maxv, minv
        if self.direction == 'horizontal':
            self.rect.set_x(minv)
            self.rect.set_width(maxv - minv)
        else:
            self.rect.set_y(minv)
            self.rect.set_height(maxv - minv)

        if self.onmove_callback is not None:
            vmin = self.pressv
            if self.direction == 'horizontal':
                vmax = event.xdata or self.prev[0]
            else:
                vmax = event.ydata or self.prev[1]

            if vmin > vmax:
                vmin, vmax = vmax, vmin
            self.onmove_callback(vmin, vmax)

        if self.eventpress:
            if self.direction == 'horizontal':
                bound = self.ax.get_ybound()
                self.draw_rubberband(minv, maxv, bound[0], bound[1])
            else:
                bound = self.ax.get_xbound()
                self.draw_rubberband(bound[0], bound[1], minv, maxv)

        return False

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from numpy.random import rand

    def onselect(*args):
        print(args)

    img = rand(100, 100)
    plt.imshow(img, cmap='winter')
    #rs = RectangleSelector(plt.gca(), onselect)
    #es = EllipseSelector(plt.gca(), onselect)
    #ls = LassoSelector(plt.gca(), onselect)
    #ls = LineSelector(plt.gca(), onselect)
    ps = PaintSelector(plt.gca(), onselect, img.shape)
    #ss = SpanSelector(plt.gca(), onselect, 'horizontal')
    plt.show()
