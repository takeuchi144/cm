#!/usr/bin/env python
# coding:utf-8
import os
import numpy as np

def postprocess(args, save_dir, df_test, y_test, y_pred, e_test):

    separate_abs_sign = args.SEPARATE_ABS_SIGN
    num_classes = args.NUM_CLASSES
    offset = args.OFFSET
    
    # fn_test
    #fn_test_path = df_test.loc[:, "sample_name2"].values
    fn_test_path = df_test
    fn_test = np.array([os.path.basename(fn) for fn in fn_test_path], dtype='U256')
    
    # predict-true_test
    if separate_abs_sign == 3:
        assert len(y_pred) == len(y_test), print("Mixmatch length y_pred and y_test.")
        
        test_len = len(y_pred)
        y_test_dec = np.zeros((test_len))
        y_pred_dec = np.zeros((test_len))
        
        for i in range(len(y_pred)):
            x0 = np.argmax(y_pred[i])
            
            if (1 <= x0) and (x0 < num_classes-1):
                y0 = y_pred[i][x0]
                yp = y_pred[i][x0+1]
                ym = y_pred[i][x0-1]
                y_pred_dec[i] = x0 + (yp-ym)/(4.0*y0-2.0*yp-2.0*ym) - offset
            else:
                y_pred_dec[i] = x0 * 1.0 - offset
        
        for i in range(len(y_test)):
            x0 = np.argmax(y_test[i])
            
            if (1 <= x0) and (x0 < num_classes-1):
                y0 = y_test[i][x0]
                yp = y_test[i][x0+1]
                ym = y_test[i][x0-1]
                y_test_dec[i] = x0 + (yp-ym)/(4.0*y0-2.0*yp-2.0*ym) - offset
            else:
                y_test_dec[i] = x0 * 1.0 - offset
    elif separate_abs_sign == 1:
        y_pred_dec = np.abs(y_pred[:,0]) * np.sign(y_pred[:,1] * 2 - 1.0)
        y_test_dec = y_test[:,0]*np.sign(y_test[:,1]*2-1.0)
    else:
        y_pred_dec = y_pred.flatten()
        y_test_dec = y_test
    
    # save result
    np.savetxt(os.path.join(save_dir, 'filename_test.txt'), fn_test, fmt='%s')
    np.savetxt(os.path.join(save_dir, 'predict-true_test.txt'), np.stack([y_pred_dec.flatten(), y_test_dec.flatten()], axis=1))
    np.savetxt(os.path.join(save_dir, 'error_bestfocus_positions.txt'), e_test)
    
    error_bfp_test = e_test
    error_bfp_pred_test = (np.abs(y_pred_dec.flatten() - y_test_dec.flatten())*2.0 > 100.0)
    print("error_rate_org: ", np.sum(error_bfp_test)/len(error_bfp_test))
    print("error_rate_pred: ", np.sum(error_bfp_pred_test)/len(error_bfp_pred_test))
    print("y_pred_shape: ",y_pred_dec.shape)
    print("y_test_shape: ",y_test_dec.shape)
