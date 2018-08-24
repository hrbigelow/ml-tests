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
    rands = np.random.randint(1, args.max_input_val, 1000)

    print('\t'.join(['---FILT---', 'INV', 'ILEN', 'OLEN', 'STRIDE',
        'DIL', 'PAD', 'MATCH', 'CMD']))

    combos = []
    for fil_str in args.filters:
        filt = [int(i) for i in fil_str.split(',')]
        filt_sz = len(filt)
        max_dil = (args.matrix_size - filt_sz) // (filt_sz - 1)
        # for inv in (True,):
        # for inv in (False,):
        for dil in range(1, max_dil + 1):
        # for dil in range(1, 2):
            # for api in ('Torch',):
            for api in ('Torch', 'TensorFlow'):
                for pad in ('VALID', 'SAME'):
                # for pad in ('VALID',):
                    for st in range(1, args.matrix_size - 1):
                    # for st in range(4, 6):
                        for inv in (False, True):
                            combos.append((filt, inv, st, dil, pad, api))

    for filt, inv, st, dil, pad, api in combos:
        filt_ctr = (len(filt) - 1) // 2
        ct = ctp.ConvType(args.matrix_size, filt, 'LC', st, inv, 0, dil, pad)

        matched_phase = -1
        inputs = []
        found_mc = []
        found_mask = []
        found_aug_input = []
        found_conv = []
        found_cmd = []

        for ph in range(ct.stride):
            ct.phase = ph
            input_sz = ct.input_size()
            if input_sz == 0:
                continue

            input = rands[0:input_sz] 
            processed_input, mc_raw, mc, mask = ct.conv(input)

            if 0 not in mask:
                continue

            if api == 'Torch':
                conv, cmd = torch_conv.conv(input, mask, filt, inv, st, ph, pad, dil)
            else:
                conv, cmd = tf_conv.conv(input, args.matrix_size, filt, inv, st, pad, dil)

            inputs.append(input)
            found_mask.append(mask)
            found_mc.append(mc)
            found_conv.append(conv)
            found_cmd.append(cmd)

            if inv:
                bool_mask = mask == 0
                aug_input = ctp.un_mask(input, bool_mask)
                found_aug_input.append(aug_input)

            if cmd.startswith('Not executed'):
                continue

            if not np.all(conv == mc):
                continue

            matched_phase = ph
            break

        # if True:
        if False:
        # if matched_phase == -1:
            print('\n')
            print('Inputs: ')
            print('\n'.join(map(str, inputs)))
            for i, (c, m) in enumerate(zip(found_conv, found_mc)):
                print('Conv {}: {}\nMatM {}: {}\n'.format(i, c, i, m))
            print('Found cmd: ')
            print('\n'.join(map(str, found_cmd)))
            print('Found masks: ')
            print('\n'.join(map(str, found_mask)))
            print('Found augi: ')
            print('\n'.join(map(str, found_aug_input)))

        print('\t'.join(map(str, [ct.filter(False).astype(int), inv, input_sz,
            len(conv), st, dil, pad, matched_phase, api, cmd])))

