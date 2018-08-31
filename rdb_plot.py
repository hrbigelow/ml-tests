# Tool for plotting RDB data in the form
# [x, y, size, color, marker]
# [x, y, size, text, orientation]

# Will use matplotlib.pyplot.{scatter,text} to implement this.
# color is an integer, and a colormap is provided by the caller
# shape is defined by mp.markers.MarkerStyle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors, markers

def make_colormap(color_names, map_name):
    color_map = [
            (i/(len(color_names) - 1),
                colors.to_rgba(colors.XKCD_COLORS[n])) for
            i, n in enumerate(color_names)
            ]
    return colors.LinearSegmentedColormap.from_list(map_name, color_map)

def make_plot(data, text, cmap, n_colors, padding_string):
    fig = plt.figure()
    #ax = plt.gca()
    padding = list(map(float, padding_string.split(',')))

    for d in data:
        x, y, s, c = d[:-2].astype('f')
        m, f = d[4:]
        col = cmap(c / (n_colors - 1))
        if f == 'none': mfc = 'none'
        else: mfc = col 
        plt.plot(x, y, m, color=col, markeredgecolor=col, markersize=s, mfc=mfc)

    for t in text:
        x, y, s = t[0:3].astype('f')
        txt = t[3]
        r = t[4].astype('f')
        ha = t[5]
        va = t[6]
        plt.text(x, y, s=txt, size=s, rotation=r, ha=ha, va=va)

    plt.axis('off')

    # Since this is a unit grid, add pad to each border 
    v = list(plt.axis())
    print(v)
    v[0] -= padding[0]
    v[1] += padding[1]
    v[2] -= padding[2]
    v[3] += padding[3]
    plt.axis(v)

    #plt.axes().set_aspect('equal')
    #plt.autoscale()
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=0)
    return fig

def parse_data_file(data_file):
    data = np.empty([0,6])
    with open(data_file, 'r') as fp:
        for dl in fp.readlines():
            data = np.append(data, [dl.strip().split('\t')], axis=0)
    return data

def parse_text_file(text_file):
    text = np.empty([0,7])
    with open(text_file, 'r') as fp:
        for tl in fp.readlines():
            text = np.append(text, [tl.strip().split('\t')], axis=0)
    return text

def parse_color_names(names_file):
    color_names = [] 
    with open(names_file, 'r') as fp:
        for cl in fp.readlines():
            color_names.append(cl.strip())
    return color_names


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='RDB Plots')
    parser.add_argument('--data_file', '-d', type=str, metavar='DATA_FILE', 
            help='Data RDB file to plot: <x>\t<y>\t<size>\t<color>\t<marker>')
    parser.add_argument('--text_file', '-t',  type=str, metavar='TEXT_FILE',
            help='File with text to plot: <x>\t<y>\t<text>\t<orientation>')
    parser.add_argument('--fig_width', '-fw', type=float, metavar='FLOAT', 
            help='Figure width in inches', default=5.0)
    parser.add_argument('--fig_height', '-fh', type=float, metavar='FLOAT', 
            help='Figure height in inches', default=5.0)
    parser.add_argument('--padding', '-pad', type=str, metavar='PADDING', 
            help='padding to add (in inches) for left,right,top,bottom', default='0.5,0.5,0.5,0.5')
    parser.add_argument('colors_file', type=str, metavar='COLORS_FILE', 
            help='File with a list of XKCD color names')
    parser.add_argument('plot_file', type=str, metavar='PLOT_OUT_FILE', 
            help='Output file for writing plot')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    if args.colors_file:
        color_names = parse_color_names(args.colors_file)

    data = []
    if args.data_file:
        data = parse_data_file(args.data_file)

    text = []
    if args.text_file:
        text = parse_text_file(args.text_file)

    
    cmap = make_colormap(color_names, 'xkcd')
    fig = make_plot(data, text, cmap, len(color_names), args.padding)
    fig.set_size_inches(args.fig_width, args.fig_height)
    fig.savefig(args.plot_file, dpi=fig.dpi)
    plt.close()



