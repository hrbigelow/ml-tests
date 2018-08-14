import conv
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def make_colormap(color_names, map_name):
    color_map = [
            (i/(len(color_names) - 1),
                colors.to_rgba(colors.XKCD_COLORS[n])) for
            i, n in enumerate(color_names)
            ]
    return colors.LinearSegmentedColormap.from_list(map_name, color_map)


if __name__ == '__main__':

    filt = [1,2,3,4,5]
    input_sz = 12 

    convs = [
            conv.ConvType(filt, 2, stride=1, is_inverse=False, phase=0, dilation=0, lpad=0, rpad=0),
            conv.ConvType(filt, 2, stride=2, is_inverse=False, phase=0, dilation=0, lpad=0, rpad=0),
            conv.ConvType(filt, 2, stride=2, is_inverse=False, phase=1, dilation=0, lpad=0, rpad=0),
            conv.ConvType(filt, 2, stride=1, is_inverse=False, phase=0, dilation=1, lpad=0, rpad=0),
            conv.ConvType(filt, 2, stride=1, is_inverse=False, phase=0, dilation=0, lpad=2, rpad=1),
            conv.ConvType(filt, 2, stride=1, is_inverse=False, phase=0, dilation=0, lpad=0, rpad=0),
            conv.ConvType(filt, 2, stride=3, is_inverse=True, phase=0, dilation=0, lpad=2, rpad=2),
            ]

    zero_color = [ 'xkcd:silver' ]
    filter_colors = [
            'xkcd:raw sienna',
            'xkcd:marigold',
            'xkcd:dark grass green',
            'xkcd:denim blue',
            'xkcd:red orange'
            ]
    mask_names = ['xkcd:silver', 'xkcd:black']

    filter_names = zero_color + filter_colors


    def make_plot(ct, path, show_in_out):
        small_space = 1.0 

        last_row_x = input_sz - 1
        # M * I = O
        times_symbol_x = last_row_x + small_space
        in_x = times_symbol_x + small_space
        equal_symbol_x = in_x + small_space
        out_x = equal_symbol_x + small_space

        mid_y = (input_sz - 1) / 2

        filter_cmap = make_colormap(filter_names, 'filter')
        data_cmap = make_colormap(mask_names, 'data')  

        mat, mask = ct.conv_mat(input_sz)
        flat = np.array([(r,c,v) for r,a in enumerate(mat) for c,v in enumerate(a)])
        data_point_sz = 220

        fig = plt.figure()
        plt.scatter(flat[:,1], flat[:,0], c=flat[:,2], s=data_point_sz, cmap=filter_cmap)

        if show_in_out:
            # *
            plt.text(times_symbol_x, mid_y, s='*', size=20, ha='center', va='baseline')

            def mask_fn(i):
                if ct.is_inverse: return (i % ct.stride) == 0
                else: return 1

            # I
            in_vec = np.array([(in_x, i, mask_fn(i)) for i in range(input_sz)])
            out_vec = np.array([(out_x, i, int(v == 0)) for i,v in enumerate(mask)])
            dvec = np.concatenate([in_vec, out_vec])

            # very strange.  If I plot in_vec and out_vec individually, the colormap
            # is inverted seemingly randomly from one call to the next.  Concatenation
            # seems to solve this, but it is likely just luck.
            plt.scatter(dvec[:,0], dvec[:,1], c=dvec[:,2], s=data_point_sz, cmap=data_cmap)
            # =
            plt.text(equal_symbol_x, mid_y, s='=', size=20, ha='center', va='baseline')

        plt.axis('off')
        plt.axes().set_aspect('equal')
        plt.autoscale()
        plt.gca().invert_yaxis()
        plt.tight_layout(pad=0)
        print('Saving {}'.format(path))
        fig.savefig(path, dpi=fig.dpi)
        plt.close()

    for i,ct in enumerate(convs):
        fname = 'mat_s{}_ph{}_inv{}_lp{}_rp{}_d{}.png'.format(
                ct.stride, ct.phase, ct.is_inverse, ct.lpad(), ct.rpad(), ct.dilation)
        make_plot(ct, fname, True) 

