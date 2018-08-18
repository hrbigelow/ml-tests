import tensorflow as tf
import convtype as ctp
import torch_conv
import tf_conv
import numpy as np
import itertools
from sys import stderr

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convolution Test')
    parser.add_argument('--matrix-size', '-sz', type=int, metavar='INT',
            help='Size of the matrix side for the convolution',
            default=50)
    parser.add_argument('--strides', '-st', nargs='+', type=int, metavar='INT',
            help='Stride for the convolution.  If --is-inverse, this is'
            ' the input stride', default=1)
    parser.add_argument('--dilations', '-dil', nargs='+', type=int, metavar='INT',
            help='Dilation for the filter', default=[1])
    parser.add_argument('--filters', '-fil', nargs='+', type=str, metavar='STR',
            help='comma-separated list of positive integers, '
            'to be used as the convolution filter')
    parser.add_argument('--paddings', '-pad', nargs='+', type=str, metavar='STR',
            help='padding for both left and right')
    parser.add_argument('--inverses', '-inv', nargs='+', type=int, metavar='INT',
            help='enter 0 for normal convolution, 1 for inverse, or both')
    parser.add_argument('--max-input-val', '-m', type=int, metavar='INT',
            help='maximum value in input cells')
    parser.add_argument('--ref-indices', '-ref', nargs='+', type=int, metavar='INT',
            help='filter reference index')
    parser.add_argument('--apis', '-ap', nargs='+', type=str, metavar='STR',
            help='\'Torch\' and/or \'TensorFlow\'')
    parser.add_argument('--phases', '-ph', nargs='+', type=int, metavar='INT',
            help='Phase overrides')

    return parser.parse_args()



if __name__ == '__main__':
    tf.enable_eager_execution()
    args = get_args()
    np.set_printoptions(linewidth=250)

    sdpi = itertools.product(args.inverses, args.strides,
            args.dilations, args.paddings, args.ref_indices, args.filters,
            args.phases)
    
    print('\t'.join(['FILT', 'INV', 'ILEN', 'OLEN', 'STRIDE',
        'DIL', 'PAD', 'REF', 'PHASE', 'MATCH', 'API', 'CMD_STRING']))

    for inv, st, dil, pad, ref, fil, phase in sdpi: 
        filt = [int(i) for i in fil.split(',')]

        # calculate phase of the first valid position of the filter after pad
        # ref_dilated = ctp.dilate_index(ref, dil)
        # used_pad = min(pad, ref_dilated)
        # phase = (ref_dilated - used_pad) % st 
        if pad not in ('VALID', 'SAME'):
            pad = (int(pad), int(pad))

        ct = ctp.ConvType(args.matrix_size, filt, ref, st, inv, phase, dil, pad)
        input_sz = ct.input_size()
        #print('params: ', ref, st, inv, phase, dil, pad)
        #print('mask: ', ct.mask())
        input = np.random.randint(1, args.max_input_val, input_sz)

        if ct.bad_input():
            continue


        processed_input, mc_raw, mc, mask = ct.conv(input)

        if 'Torch' in args.apis:
            trc, tr_cmd = torch_conv.conv(ct, input)
            tr_same = np.all(trc == mc)
            if not tr_same:
                print('trc: ', trc)
                print('mmc: ', mc)

            print('\t'.join(map(str, [ct.filter(False).astype(int), inv, input_sz,
                len(trc), st, dil, pad, ref, phase, tr_same, 'Torch', tr_cmd])))

        if 'TensorFlow' in args.apis:
            if dil > 1:
                print('conv1d_transpose doesn\'t support dilation > 1', file=stderr)
                continue

            if ct.padding_type() not in ('VALID', 'SAME'):
                print('ConvType must be VALID or SAME to compare with TensorFlow convolution',
                        file=stderr)
            else:
                with tf.device('/cpu:0'):
                    tfc, tf_cmd = tf_conv.conv(ct, input)
                tf_same = np.all(tfc == mc)
                if not tf_same:
                    print('inp: ', input)
                    print('tfc: ', tfc)
                    print('mmc: ', mc)
                print('\t'.join(map(str, [ct.filter(False).astype(int), inv, input_sz,
                    len(tfc), st, dil, pad, ref, phase, tf_same, 'TensorFlow', tf_cmd])))

