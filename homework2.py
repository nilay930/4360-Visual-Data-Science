# Sources:
# Used chatgpt to tell me how to select certain rows and columns. 
# ie. How do I select last 2 columns of 2d np array 

import argparse
import numpy as np
from scipy.linalg import pinv, svd # Your only additional allowed imports!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Homework 2",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2023",
        add_help = "How to use",
        prog = "python homework2.py <arguments>")
    parser.add_argument("-f", "--infile", required = True,
        help = "Dynamic texture file, a NumPy array.")
    parser.add_argument("-q", "--dimensions", required = True, type = int,
        help = "Number of state-space dimensions to use.")
    parser.add_argument("-o", "--output", required = True,
        help = "Path where the 1-step prediction will be saved as a NumPy array.")

    args = vars(parser.parse_args())

    # Collect the arguments.
    input_file = args['infile']
    q = args['dimensions']
    output_file = args['output']

    # Read in the dynamic texture data.
    M = np.load(input_file)
    
    f = M.shape[0]
    h = M.shape[1]
    w = M.shape[2]

    M2 = M.reshape(f,h*w)
    Y = M2.T

    # SVD for X and C
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)

    # Appearance Model y_t = Cx_t + u_t (slide 12)

    # C
    C = U[:, :q]

    # X
    Sig_hat = np.diag(s[:q])
    Vh_hat = Vh[:q, :] 

    X = Sig_hat @ Vh_hat

    # A
    X1_fm1 = X[:, :-1] #X_1^(f-1)
    X2_f = X[:, 1:] # X_2^f

    X1_fm1_pinv = np.linalg.pinv(X1_fm1) 
    A = X2_f @ X1_fm1_pinv

    # Step sim

    x_f = X[-q:,-1]  
    #x_f = X[:,-1]
    #x_f = x_f.reshape(q,1)  

    x_fp1 = A @ x_f
    y_fp1 = C @ x_fp1

    #print(y_fp1.shape)
    y_fp1_reshaped = y_fp1.reshape(h, w)
    #print(y_fp1_reshaped.shape)

    print("A ", A.shape)
    print("y",Y.shape)
    print("s",s.shape)
    print("c",C.shape)
    print("x1", X1_fm1.shape)
    print("x2", X2_f.shape)
    print("out", y_fp1_reshaped.shape)

    print(y_fp1_reshaped)

    print("done")
    np.save(output_file, y_fp1_reshaped)
