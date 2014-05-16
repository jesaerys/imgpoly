"""

=========
`imgpoly`
=========

Determine footprints for FITS images.

This package is centered on the `find_borders` function, which calculates
the border of an input boolean array using a technique inspired by the
`marching squares algorithm <wikipedia.org/wiki/Marching_squares>`_. This
is particularly useful for astronomical CCD images, which often have a
"vignette" of non-data pixels around the edge.

A small set of helper functions is provided, each tailored to FITS images
from a particular instrument. HST/ACS/WFC, HST/WFC3, and GALEX are
currently supported. The main difference between analyses for any two
instruments is how the boolean arrays (data pixels vs. border pixels) for
`find_borders` are constructed as different detectors have different
shapes.

`imgpoly` depends on the following packages:

- `astropy <http://www.astropy.org>`_
- `numpy <http://www.numpy.org>`_
- `shapely <https://github.com/Toblerity/Shapely>`_
- `geoutil <https://github.com/jesaerys/geoutil>`_


Functions
---------

========================= ================================================
`find_borders`            Border tracing algorithm.
`acs_poly`                Determine the chip outlines of an ACS FITS image.
`wfc3_poly`               Determine the chip outline of a WFC3 FITS image.
`galex_poly`              Determine the chip outline of a GALEX FITS image.
========================= ================================================

.. rubric:: Private functions

========================= ================================================
`_build_stepdict`         Return an step direction dictionary.
`_simplify_poly`          Reduce the number of points that define a
                          polygon.
`_convert_to_pixelcoords` Convert polygon vertices from slice coordinates
                          to pixel coordinates.
`_area`                   Return the area of a polygon.
`_simplify_acs_poly`      Extract the main outline of an ACS chip polygon.
========================= ================================================

"""
from collections import OrderedDict

from astropy.io import fits
import geoutil
import numpy as np
from shapely import geometry


def _build_stepdict():
    """Return an step direction dictionary.

    Build lookup table linking unique sums (integers from 1 to 14) and
    previous step directions to the direction of the next step. Step
    directions are given as (dy, dx). If there is no relevant last step,
    use None.

    Returns
    -------
    out : dict
        Dictionary of directions as (dy, dx), keyed by (sum, last_step).

    Notes
    -----
    Notions of north and south can be confusing when considering how arrays
    are printed in a terminal, where line numbers increase further down the
    screen. North is defined to be the direction of increasing row number,
    such that col,row in an array corresponds directly with x,y in a graph.

    """
    N, E, S, W = (1, 0), (0, 1), (-1, 0), (0, -1)  # (drow, dcol)

    stepdict = {(1, None): N,  (2, None): E,  (3, None): E,  (4, None): W,
                (5, None): N,  (6, None): W,  (7, None): E,  (8, None): S,
                (9, None): N, (10, None): S, (11, None): S, (12, None): W,
                (13, None): N, (14, None): W,
                (1, E): N,  (2, S): E,  (3, E): E,  (4, N): W,
                (5, N): N,  (6, N): W,  (6, S): E,  (7, N): E,
                (8, W): S,  (9, E): N,  (9, W): S, (10, S): S,
                (11, E): S, (12, W): W, (13, W): N, (14, S): W}
    return stepdict


def _simplify_poly(poly):
    """Reduce the number of points that define a polygon.

    Simplify a polygon by removing points that lie on straight lines,
    leaving only the vertices that define corners.

    Parameters
    ----------
    poly : 2-dimensional np.array
        nx2 array of coordinates.

    Returns
    -------
    out : 2-dimensional np.array
        nx2 array of coordinates, less any vertices that were on straight
        lines in poly.

    """
    poly = np.vstack((poly[-1], poly, poly[0]))
    compare_prev = poly[1:-1] == poly[:-2]
    compare_next = poly[1:-1] == poly[2:]
    compare = compare_prev & compare_next
    compare = compare[:, 0] | compare[:, 1]
    return poly[1:-1][-compare]


def _convert_to_pixelcoords(poly_list):
    """Convert polygon vertices from slice coordinates to pixel
    coordinates.

    The slice coordinate system assigns integers to the lines between
    pixels in an image. Coordinate pairs are given in row,col order, and
    the outside corner of the ``image_arr[0, 0]`` pixel is at (0, 0). The
    pixel coordinate system assigns integers to the centers of pixels in an
    image. Coordinate pairs are given in x,y (i.e. col,row) order, and the
    outside corner of the ``image_arr[0, 0]`` pixel is at (0.5, 0.5).

    Parameters
    ----------
    poly_list : list
        List of polygons specified as nx2 arrays of slice coordinates in
        row,col order.

    Returns
    -------
    out : list
        List of polygons in poly_list with coordinates converted to pixel
        coordinates in x,y (i.e. col,row) order.

    """
    return [poly[:, ::-1] + 0.5 for poly in poly_list]


def find_borders(boolarr):
    """Border tracing algorithm.

    Find the polygons that trace the borders in binary data, using a method
    inspired by the 2D marching squares algorithm; see below for a detailed
    explanation.

    Parameters
    ----------
    boolarr : 2-dimensional `np.array`
        2D array containing binary data (0's and 1's, or True's and
        False's).

    Returns
    -------
    out : list
        List of polygons specified in pixel coordinates as nx2 arrays.
        Polygon vertices are listed in counter-clockwise order, except if
        the polygon is actually a hole, in which case the vertices are
        listed in clockwise order.

    Notes
    -----
    The input image is used to form a boolean array, equal to 1 where
    pixels contain valid data and 0 where invalid. The interface between
    0's and 1's forms the border to be traced, and is thus defined by a set
    of vertices (points where four pixels meet) in the boolean array.
    Starting at one such vertex, the border can be traced out by stepping
    through successive vertices while always keeping 1's on the left (with
    respect to the direction of traversal) until the original vertex is
    encountered, forming a closed loop. The details for deciding the
    direction of the next vertex in each iteration are described below.

    There are 16 possibilities for the values of the pixels around each
    vertex. By assigning each pixel in the group a binary flag (0, 1, 2, 4,
    or 8), the pixel configuration around any vertex can be uniquely
    determined from the sum of the binary flags. Non-zero pixels around a
    vertex are flagged using the following pattern (0's in the boolean
    array are flagged as 0)::

      y---+---+
      |(1)|(2)|
      +---+---+
      |(4)|(8)|
      o---+---x

    The possible configurations, their sums, and the next direction to go
    in each case (using the 1's-on-the-left convention), are as follows::

      y---+---+ y---+---+ y---+---+ y---+---+ y---+---+ y---+---+ y---+---+
      | 1 | 0 | | 0 | 1 | | 1 | 1 | | 0 | 0 | | 1 | 0 | | 0 | 1 | | 1 | 1 |
      +---+---+ +---+---+ +---+---+ +---+---+ +---+---+ +---+---+ +---+---+
      | 0 | 0 | | 0 | 0 | | 0 | 0 | | 1 | 0 | | 1 | 0 | | 1 | 0 | | 1 | 0 |
      o---+---x o---+---x o---+---x o---+---x o---+---x o---+---x o---+---x
        1   N     2   E     3   E     4   W     5   N     6   W     7   E

      y---+---+ y---+---+ y---+---+ y---+---+ y---+---+ y---+---+ y---+---+
      | 0 | 0 | | 1 | 0 | | 0 | 1 | | 1 | 1 | | 0 | 0 | | 1 | 0 | | 0 | 1 |
      +---+---+ +---+---+ +---+---+ +---+---+ +---+---+ +---+---+ +---+---+
      | 0 | 1 | | 0 | 1 | | 0 | 1 | | 0 | 1 | | 1 | 1 | | 1 | 1 | | 1 | 1 |
      o---+---x o---+---x o---+---x o---+---x o---+---x o---+---x o---+---x
        8   S     9   N    10   S    11   S    12   W    13   N    14   W

    The fully-transparent and fully-opaque cases (the 15th and 16th
    configurations) do not contain the border, so they are not involved in
    tracing process.

    The sum=6 and sum=9 cases are ambiguous: the direction for sum=6 could
    be east or west, and north or south for sum=9 (the directions given
    above are defaults). The ambiguities are resolved by imposing a
    left-turn rule. For sum=6, move west if the last step was north, move
    east if the last step was south. For sum=9, move south if the last step
    was west, move north if the last step was east.

    Note that another way to solve the sum=6 and sum=9 direction
    ambiguities is to make left turns in the sum=6 case and to make right
    turns in the sum=9 case (or the other way around, it doesn't matter).
    This results in less-fragmented polygons because the sum=9 points
    effectively serve as bridges between pixels that would belong to
    separate polygons under the left-turn-only convention. However, the
    sum=9 points are also places where the polygons pinch in on themselves;
    such polygons are generally not considered geometrically valid. It is
    much easier to deal with polygons generated from the left-turn-only
    system because they are guaranteed to be free of pinch points, even if
    the convention generates a larger number of smaller polygons.

    Also note that some polygons returned by `find_borders` may actually be
    holes (polygon cutouts of larger polygons). There are two issues that
    arise when a hole is assumed to be a polygon: 1) the polygon lies
    inside of another, which doesn't make sense, and 2) such a polygon may
    contain pinch points. Holes can be easily identified because their
    vertices are listed in clockwise order, whereas the vertices of solid
    polygons are listed in counter-clockwise order, because of the
    1's-on-the-left and left-turn conventions discussed above.

    The sum for each vertex is placed in an array which represents the
    "topology" of the boolean array. Vertices for the edge and corner
    pixels are handled as if the boolean array was surrounded by zeros. A
    list of border vertices (those with sums > 0 and < 15) is compiled, and
    all borders in the image are traced and recorded until all vertices in
    the list have been visited.

    """
    stepdict = _build_stepdict()

    boolarr_rows, boolarr_cols = boolarr.shape
    sumarr = np.zeros((boolarr_rows+1, boolarr_cols+1), 'int')
    sumarr[:-1, 1:] += boolarr
    sumarr[:-1, :-1] += 2 * boolarr
    sumarr[1:, 1:] += 4 * boolarr
    sumarr[1:, :-1] += 8 * boolarr

    # Indices of all vertices that trace the border. Note that vertices
    # with sum=6 and sum=9 are included in the points list twice so that
    # both direction choices are explored.
    points = np.vstack((np.c_[np.where((0 < sumarr) & (sumarr < 15))],
                        np.c_[np.where(sumarr == 6)],
                        np.c_[np.where(sumarr == 9)])).tolist()
    premove = points.remove

    poly_list = []
    while len(points):
        i0, j0 = points[0]
        vertices = []
        vappend = vertices.append
        dr_last = None

        premove([i0, j0])
        vappend((i0, j0))
        s = sumarr[i0, j0]
        dr = stepdict[(s, dr_last)]
        i, j, dr_last = i0 + dr[0], j0 + dr[1], dr

        # Reverse direction if next point has already been covered.
        if (s == 6) and ([i, j] not in points):
            i, j = i0 + dr[0], j0 - dr[1]
            dr_last = (dr[0], -dr[1])
        if (s == 9) and ([i, j] not in points):
            i, j = i0 - dr[0], j0 + dr[1]
            dr_last = (-dr[0], dr[1])

        while (i, j) != (i0, j0):
            premove([i, j])
            vappend((i, j))
            s = sumarr[i, j]
            dr = stepdict[(s, dr_last)]
            i, j, dr_last = i + dr[0], j + dr[1], dr

        poly_list.append(np.array(vertices))
    poly_list = [_simplify_poly(poly) for poly in poly_list]
    poly_list = _convert_to_pixelcoords(poly_list)
    return poly_list


def _area(poly):
    """Return the area of a polygon.

    Area is calculated by adding up trapezoids formed by all pairs of
    adjacent vertices, going around the polygon in a uniform direction so
    that some trapezoids add to the area and some subtract. The sign of the
    area is positive if the polygon vertices are listed in a
    counter-clockwise order and negative if clockwise.

    Parameters
    ----------
    poly : 2-dimensional np.array
        A polygon specified as an nx2 array of x,y coordinates.

    Returns
    -------
    out : float
        The area of the polygon, positive if the polygon is
        counter-clockwise, negative if clockwise.

    """
    poly = np.vstack((poly, poly[0:1]))  # Wrap end around to first point.
    x_list, y_list = poly.T
    x_list, y_list = x_list - np.min(x_list), y_list - np.min(y_list)
    dx, dy = x_list[1:] - x_list[:-1], y_list[1:] - y_list[:-1]
    y0 = np.min(np.c_[y_list[:-1], y_list[1:]], axis=1)
    return -np.sum(dx*y0 + dx*dy/2.)


def _simplify_acs_poly(poly):
    """Extract the main outline of an ACS chip polygon.

    Polygons for outlines of ACS chips can contain many artifacts that
    confuse the main chip outline. This function forms a polygon from the
    main chip outline ignoring all holes, which is especially useful for
    plotting.

    Parameters
    ----------
    poly : `Polygon` or `MultiPolygon` from `shapely.geometry` or None
        A polygon representing an ACS chip in full detail (including holes
        and islands).

    Returns
    -------
    out : `Polygon` or `MultiPolygon` from `shapely.geometry` or None
        A hole-less polygon for the main outline of an ACS chip.

    """
    # Identify the largest polygon.
    if poly is None:
        return None
    elif poly.type == 'Polygon':
        poly = [poly]
    area_list = [subpoly.area for subpoly in poly]
    i = np.argmax(area_list)
    poly = poly[i]

    # Only copy the exterior to get rid of any holes.
    xy = poly.exterior.coords
    poly = geometry.Polygon(xy)
    return poly


def acs_poly(img_file, coordsys='pixel'):
    """Determine the chip outlines of an ACS FITS image.

    `find_borders` is used to find the borders around all valid data
    pixels, i.e. those not equal to the value of a characteristic non-data
    pixel. The chips are identified as the two border polygons with the
    largest areas, and are returned as a `geoutil.Geoset` in either pixel
    or world coordinates. The image data is assumed to be in the first FITS
    extension.

    Parameters
    ----------
    img_file : str
        Path to the ACS FITS image.
    coordsys : {'pixel', 'WCS'}, optional
        Set the coordinate system of the polygons. Default value is
        'pixel'. If 'WCS', then the coordinates are transformed according
        to the WCS information in the FITS header of the image.

    Returns
    -------
    out : `geoutil.Geoset`
        A `geoutil.Geoset` instance containing the chip outline polygons.

    """
    EXT = 1  # Science data extension for ACS FITS images.

    hdulist = fits.open(img_file)
    img, hdr = hdulist[EXT].data, hdulist[EXT].header

    xrep, yrep = -5, 5  # Representative non-data pixel.
    non_data = img[yrep, xrep]

    boolarr = np.where(img == non_data, False, True)
    poly_list = find_borders(boolarr)

    # Find the largest polygon from the find_borders function:
    area_list = [_area(poly) for poly in poly_list]
    i1, i2 = np.argsort(area_list)[-2:]
    chip1 = poly_list[i1]
    chip1 = geometry.Polygon(chip1.tolist())
    chip1 = geoutil.validate_poly(chip1)
    chip2 = poly_list[i2]
    chip2 = geometry.Polygon(chip2.tolist())
    chip2 = geoutil.validate_poly(chip2)

    # Convert coordinates, if specified:
    if coordsys == 'WCS':
        chip1 = geoutil.poly_pix2world([chip1], hdr)[0]
        chip2 = geoutil.poly_pix2world([chip2], hdr)[0]

    # Build geoset:
    geo1 = geoutil.Geo(chip1, attrs=OrderedDict([('chip', 1)]))
    geo2 = geoutil.Geo(chip2, attrs=OrderedDict([('chip', 2)]))
    item = geoutil.Item([geo1, geo2])
    attrs = OrderedDict([('source', img_file.split('/')[-1]),
                         ('description', 'HST/ACS image outline'),
                         ('coordsys', coordsys)])
    geoset = geoutil.Geoset(item, attrs=attrs, hdr=hdr)
    return geoset


def wfc3_poly(img_file, coordsys='pixel'):
    """Determine the chip outline of a WFC3 FITS image.

    `find_borders` is used to find the borders around either 1) all pixels
    where the corresponding context ('CTX') image is greater than zero, or,
    2) if a context image is not available, all valid data pixels, i.e.,
    those not equal to the value of a characteristic non-data pixel. The
    chip is identified as the border polygon with the largest area, and is
    returned as a `geoutil.Geoset` in either pixel or world coordinates.
    The image data is assumed to be in the first FITS extension. The
    context image, which is zero where there are no exposures, is assumed
    to be in the third FITS extension.

    Parameters
    ----------
    img_file : str
        Path to the WFC3 FITS image.
    coordsys : {'pixel', 'WCS'}, optional
        Set the coordinate system of the polygons. Default value is
        'pixel'. If 'WCS', then the coordinates are transformed according
        to the WCS information in the FITS header of the image.

    Returns
    -------
    out : `geoutil.Geoset`
        A `geoutil.Geoset` instance containing the chip outline polygon.

    """
    SCIEXT = 1  # Science data extension for WFC3 FITS images
    CTXEXT = 3  # Context data extension for WFC3 FITS images

    hdulist = fits.open(img_file)
    try:
        img, hdr = hdulist[CTXEXT].data, hdulist[SCIEXT].header
        non_data = 0  # Value of non-data pixels in CTX image.
    except IndexError:
        img, hdr = hdulist[SCIEXT].data, hdulist[SCIEXT].header
        xrep, yrep = -5, 5  # Representative non-data pixel.
        non_data = img[yrep, xrep]

    boolarr = np.where(img == non_data, False, True)
    poly_list = find_borders(boolarr)

    # Find the largest polygon from the find_borders function:
    area_list = [_area(poly) for poly in poly_list]
    i = np.argmax(area_list)
    chip = poly_list[i]
    chip = geometry.Polygon(chip.tolist())
    chip = geoutil.validate_poly(chip)

    # Convert coordinates, if specified:
    if coordsys == 'WCS':
        chip = geoutil.poly_pix2world([chip], hdr)[0]

    # Build geoset:
    geo = geoutil.Geo(chip)
    item = geoutil.Item(geo)
    attrs = OrderedDict([('source', img_file.split('/')[-1]),
                         ('description', 'HST/WFC3 image outline'),
                         ('coordsys', coordsys)])
    geoset = geoutil.Geoset(item, attrs=attrs, hdr=hdr)
    return geoset


def galex_poly(img_file, coordsys='pixel'):
    """Determine the chip outline of a GALEX FITS image.

    `find_borders` is used to find the border polygons around all valid
    data pixels, i.e. those not equal to the value of a characteristic
    non-data pixel. The chip is identified as the border polygon with the
    largest area, and is returned as a `geoutil.Geoset` in either pixel or
    world coordinates. The image data is assumed to be in the primary FITS
    hdu.

    It is highly-recommended to use *rrhr.fits (high-resolution relative
    response) images rather than actual data images. Data images can have
    valid pixels equal to the non-data value, which can make the border
    much more complicated and messy, and thus longer to calculate.

    Parameters
    ----------
    img_file : str
        Path to the GALEX FITS image.
    coordsys : {'pixel', 'WCS'}, optional
        Set the coordinate system of the polygons. Default value is
        'pixel'. If 'WCS', then the coordinates are transformed according
        to the WCS information in the FITS header of the image.

    Returns
    -------
    out : `geoutil.Geoset`
        A `geoutil.Geoset` instance containing the chip outline polygon.

    """
    EXT = 0  # Science data extension for GALEX FITS images

    hdulist = fits.open(img_file)
    img, hdr = hdulist[EXT].data, hdulist[EXT].header

    xrep, yrep = 300, 300  # Representative non-data pixel.
    non_data = img[yrep, xrep]

    boolarr = np.where(img == non_data, False, True)
    poly_list = find_borders(boolarr)

    # Find the largest polygon from the find_borders function:
    area_list = [_area(poly) for poly in poly_list]
    i = np.argmax(area_list)
    chip = poly_list[i]
    chip = geometry.Polygon(chip.tolist())
    chip = geoutil.validate_poly(chip)

    # Convert coordinates, if specified:
    if coordsys == 'WCS':
        chip = geoutil.poly_pix2world([chip], hdr)[0]

    # Build geoset:
    geo = geoutil.Geo(chip)
    item = geoutil.Item(geo)
    attrs = OrderedDict([('source', img_file.split('/')[-1]),
                         ('description', 'GALEX image outline'),
                         ('coordsys', coordsys)])
    geoset = geoutil.Geoset(item, attrs=attrs, hdr=hdr)
    return geoset
