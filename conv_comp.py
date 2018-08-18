# Compare Torch convolutions with the matrix multiply approach
import convtype as ctp
import torch_conv
import numpy as np
from sys import stderr


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convolution Test')
    parser.add_argument('--matrix-size', '-sz', type=int, metavar='INT',
            help='Size of the matrix side for the convolution',
            default=50)
    parser.add_argument('--filters', '-fil', nargs='+', type=str, metavar='STR',
            help='comma-separated list of positive integers, '
            'to be used as the convolution filter')
    parser.add_argument('--max-input-val', '-m', type=int, metavar='INT',
            help='maximum value in input cells')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    np.set_printoptions(linewidth=250)

    print('\t'.join(['FILT', 'INV', 'ILEN', 'OLEN', 'STRIDE',
        'DIL', 'PAD', 'MATCH', 'CMD_STRING']))

    comps = []
    for fil_str in args.filters:
        filt = [int(i) for i in fil_str.split(',')]
        filt_sz = len(filt)
        for inv in (False, True):
            for st in range(1, args.matrix_size - 1):
                max_dil = (args.matrix_size - filt_sz) // (filt_sz - 1)
                for dil in range(1, max_dil + 1):
                    for pad in ('VALID', 'SAME'):
                        for api in ('Torch', 'TensorFlow'):
                            comps.append((filt, inv, st, dil, pad, api))

    for filt, inv, st, dil, pad, api in comps:
        if inv:
            if api == 'Torch':
                ct = torch_conv.get_ct_transpose(args.matrix_size, filt, st, pad, dil)
            else: 
                ct = tf_conv.get_ct_transpose(args.matrix_size, filt, st, pad, dil)
        else:
            if api == 'Torch':
                ct = torch_conv.get_ct(args.matrix_size, filt, st, pad, dil)
            else:
                ct = tf_conv.get_ct_transpose(args.matrix_size, filt, st, pad, dil)

        input_sz = ct.input_size()
        input = np.random.randint(1, args.max_input_val, input_sz)

        if ct.bad_input():
            continue

        processed_input, mc_raw, mc, mask = ct.conv(input)

        if api == 'Torch':
            conv, cmd = torch_conv.conv(input, filt, inv, st, pad, dil)
        else:
            conv, cmd = tf_conv.conv(input, filt, inv, st, pad, dil)
        same = np.all(conv == mc)
        if not same:
            print('input: ', input)
            print('mask : ', mask)
            print('torch: ', conv)
            print('matml: ', mc)

        print('\t'.join(map(str, [ct.filter(False).astype(int), inv, input_sz,
            len(conv), st, dil, pad, same, cmd])))

