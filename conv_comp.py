# Compare Torch convolutions with the matrix multiply approach
import tensorflow as tf
import convtype as ctp
import torch_conv
import tf_conv
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
    tf.enable_eager_execution()

    print('\t'.join(['FILT', 'INV', 'ILEN', 'OLEN', 'STRIDE',
        'DIL', 'PAD', 'MATCH', 'CMD']))

    combos = []
    for fil_str in args.filters:
        filt = [int(i) for i in fil_str.split(',')]
        filt_sz = len(filt)
        max_dil = (args.matrix_size - filt_sz) // (filt_sz - 1)
        for dil in range(1, max_dil + 1):
            for api in ('Torch', 'TensorFlow'):
                for pad in ('VALID', 'SAME'):
                    for st in range(1, args.matrix_size - 1):
                        for inv in (False, True):
                            combos.append((filt, inv, st, dil, pad, api))

    for filt, inv, st, dil, pad, api in combos:
        if inv:
            if api == 'Torch':
                ct = torch_conv.get_ct_transpose(args.matrix_size, filt, st, pad, dil)
            else: 
                ct = tf_conv.get_ct_transpose(args.matrix_size, filt, st, pad, dil)
        else:
            if api == 'Torch':
                ct = torch_conv.get_ct(args.matrix_size, filt, st, pad, dil)
            else:
                ct = tf_conv.get_ct(args.matrix_size, filt, st, pad, dil)

        assert ct.is_inverse == inv
        input_sz = ct.input_size()
        input = np.random.randint(1, args.max_input_val, input_sz)

        matched_phase = -1
        found_mc = []
        found_mask = []

        if api == 'Torch':
            conv, cmd = torch_conv.conv(input, filt, inv, st, pad, dil)
        else:
            conv, cmd = tf_conv.conv(input, args.matrix_size, filt, inv, st, pad, dil)

        for ph in range(ct.stride):
            ct.phase = ph
            processed_input, mc_raw, mc, mask = ct.conv(input)

            same = np.all(conv == mc)
            if not same:
                found_mc.append(mc)
                found_mask.append(mask)
                continue
            else:
                matched_phase = ph
                break
            #if not same:
            #    print('a, b: ', ct.ref_bounds_allowed())
            #    print('x, y: ', ct.ref_bounds_avoid_pad())
            #    print('input: ', input)
            #    print('mask : ', mask)
            #    print('{}: {}'.format(api, conv))
            #    print('matml_mc: ', mc)

        if matched_phase == -1:
            print('\n')
            print('Conv: ', conv)
            print('Found mcs:')
            print('\n'.join(map(str, found_mc)))
            print('Found masks: ')
            print('\n'.join(map(str, found_mask)))

        print('\t'.join(map(str, [ct.filter(False).astype(int), inv, input_sz,
            len(conv), st, dil, pad, matched_phase, api, cmd])))

