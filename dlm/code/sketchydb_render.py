#!/usr/bin/env python3
# coding: utf-8

'''
Rasterize all Sketchy drawings in DB.

Usage: python3 sketchydb_render.py /path/to/db.json

Authors:
- Luis A. Leiva <luis.leiva@aalto.fi>

Dependencies:
- tkinter: sudo apt-get install python3-tk
- canvasvg: pip install canvasvg
'''

import sys
import os
import json
from collections import defaultdict
import tkinter as tk
import canvasvg


def load_sketches(json_filename):
    '''Load multiple skecthes from JSON file.'''
    with open(json_filename) as f:
        dataset = json.load(f)
    return dataset


def draw_sketch(points, colors, thicks, img_width, img_height):
    '''Draw points on canvas.'''
    root = tk.Tk()
    canvas = tk.Canvas(root, width=img_width, height=img_height, bg='#ffffff')

    for j, coords in points.items():
        color = colors[j]
        thick = thicks[j]

        for i, (x1, y1) in enumerate(coords):
            if i > 0:
                (x0, y0) = coords[i - 1]
                canvas.create_line(x0, y0, x1, y1, fill=color, width=thick)

    canvas.pack()
#    root.mainloop() # for debugging only
    return canvas


def bounding_box(points):
    '''Compute bounding box of points.'''
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for (x,y) in points:
        if x < min_x: min_x = x
        if y < min_y: min_y = y
        if x > max_x: max_x = x
        if y > max_y: max_y = y

    return (min_x, min_y, max_x, max_y)


def move_points(points, x=0.0, y=0.0):
    '''Move points by given offsets.'''
    return [(px + x, py + y) for px, py in points]


def scale_points(points, x=1.0, y=1.0):
    '''Scale points by given factors.'''
    return [(px * x, py * y) for px, py in points]


def write_svg(canvas, width, height, filename):
    '''Write SVG file with the canvas data.'''
    doc = canvasvg.SVGdocument()
    for element in canvasvg.convert(doc, canvas):
        doc.documentElement.appendChild(element)

    doc.documentElement.setAttribute('width',  str(width))
    doc.documentElement.setAttribute('height', str(height))

    with open(filename, 'w') as f:
        f.write(doc.toprettyxml())


def save_drawing(sketch, prefix='', img_width=200, img_height=200, padding=20):
    '''Parse sketch data and export to file.'''
    strokes = defaultdict(list)
    colors = defaultdict(list)
    thicks = defaultdict(list)

    # Stroke points were stored as a flatten list, so fix.
    points_x = defaultdict(list)
    points_y = defaultdict(list)

    # Compute the bounding box of the sketch as a whole,
    # instead of per-stroke.
    xmin, ymin = float('inf'), float('inf')
    xmax, ymax = float('-inf'), float('-inf')

    for (j, obj) in enumerate(sketch):
        # The points data struct is [x1,y1,x2,y2,...].
        for (i, coord) in enumerate(obj['coords']):
            if i % 2 == 0:
                points_x[j].append(coord)
            else:
                points_y[j].append(coord)

        colors[j].append(obj['color'])
        thicks[j].append(obj['width'])

        strokes[j] = list(zip(points_x[j], points_y[j]))

        (min_x, min_y, max_x, max_y) = bounding_box(strokes[j])

        if min_x < xmin: xmin = min_x
        if min_y < ymin: ymin = min_y
        if max_x > xmax: xmax = max_x
        if max_y > ymax: ymax = max_y

    xmin -= padding
    ymin -= padding
    xmax += padding
    ymax += padding

    width, height = xmax - xmin, ymax - ymin

    # Avoid scaling distortions.
    if width > height: height = width
    if height > width: width = height

    # Ensure coordinates are relative to the top-left corner,
    # then scale (proportionally) to the desired output size.
    for j, points in strokes.items():
        strokes[j] = move_points(strokes[j], width/2, height/2)
        strokes[j] = scale_points(strokes[j], img_width/width, img_height/height)

    # NB: When img_{width,height} are smaller than the actual image size,
    # the output SVG will be cropped, not stretched.
    drawing = draw_sketch(strokes, colors, thicks, img_width, img_height)
    outfile = '{}.svg'.format(prefix)
    write_svg(drawing, img_width, img_height, outfile)


if __name__ == '__main__':
    json_db = sys.argv[1]
    dataset = load_sketches(json_db)
    for i, entry in enumerate(dataset):
        if i>=5:
            break

        # Out filename will be the sketch ID, assuming it's always unique.
        # TODO Shaun: check this assumption! filename=id, and the filename is always unique
        img_id = entry['_id']['$oid']
        sketch = entry['interactionData']['peekedSketch']
        save_drawing(sketch, img_id, img_width=400, img_height=400)
