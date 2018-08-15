import convtype as ctp
import torch_conv
import numpy as np
import itertools
from sys import stderr

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convolution Test')
    parser.add_argument('--input-size', '-sz', type=int, metavar='INT',
            help='Size of the input array for the convolution',
            default=50)
    parser.add_argument('--strides', '-st', nargs='+', type=int, metavar='INT',
            help='Stride for the convolution.  If --is-inverse, this is'
            ' the input stride', default=1)
    parser.add_argument('--dilations', '-di', nargs='+', type=int, metavar='INT',
            help='Dilation for the filter', default=[0])
    parser.add_argument('--filters', '-f', nargs='+', type=str, metavar='STR',
            help='comma-separated list of positive integers, '
            'to be used as the convolution filter')
    parser.add_argument('--paddings', '-pa', nargs='+', type=int, metavar='INT',
            help='padding for both left and right')
    parser.add_argument('--inverses', '-in', nargs='+', type=int, metavar='INT',
            help='enter 0 for normal convolution, 1 for inverse, or both')
    parser.add_argument('--max-input-val', '-m', type=int, metavar='INT',
            help='maximum value in input cells')
    parser.add_argument('--ref-indices', '-r', nargs='+', type=int, metavar='INT',
            help='filter reference index')

    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    input = np.random.randint(1, args.max_input_val, args.input_size)
    np.set_printoptions(linewidth=250)

    sdpi = itertools.product(args.inverses, args.strides,
            args.dilations, args.paddings, args.ref_indices, args.filters)
    
    print('\t'.join(['INV', 'FILT', 'ILEN', 'STRIDE',
        'DIL', 'PAD', 'REF', 'PHASE', 'MATCH', 'CMD_STRING']))

    for inv, st, dil, pad, ref, fil in sdpi: 
        filt = [int(i) for i in fil.split(',')]

        # calculate phase of the first valid position of the filter after pad
        ref_dilated = ctp.dilate_index(ref, dil)
        used_pad = min(pad, ref_dilated)
        phase = (ref_dilated - used_pad) % st 

        ct = ctp.ConvType(filt, ref, st, inv, phase, dil, pad, pad)
        if ct.bad_input():
            print('ConvType: bad input', file=stderr)
            continue

        if not ct.usable_padding():
            print('ConvType cannot use all padding.'
                    ' filt {}, ref {}, dil {}, pad {}'.format(filt, ref, dil, pad),
                    file=stderr)
            continue

        tc, cmd_string = torch_conv(ct, input)
        i, p, mc_raw, mask = ct.conv(input)
        mc = mc_raw[mask == ctp.VALID_VAL]
        same = np.all(tc == mc)

        # if tc.shape != mc.shape:
            # print('Error: tc: {}, mc: {}'.format(tc.shape, mc.shape))
        if not same:
            #print('Not the same:')
            #print('in: ', input)
            print('tc: ', tc)
            print('mc: ', mc)

        print('\t'.join(map(str, [inv, ct.filter(False), args.input_size,
            st, dil, pad, ref, phase, same, cmd_string])))

