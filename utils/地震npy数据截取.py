import numpy as np
import torch


if __name__ == '__main__':

    #截取kerry
    # s = np.load("D:/data/kerry3D.npy")
    # print(s.shape)
    # z = 8
    # x = 16
    # y = 41
    #
    # s = s[0+z:768+z,0+x:192+x,0+y:640+y]
    # np.save("D:/data/kerry.npy",s)
    # print(s.shape)

    # 截取PCB
    # s = np.load("D:/data/PCB10011008700.npy")
    # print(s.shape)
    # z = 0
    # x = 0
    # y = 0
    #
    # s = s[0 + z:768 + z, 0 + x:1088 + x, 0 + y:448 + y]
    # np.save("D:/data/PCB.npy", s)
    # print(s.shape)

    s = np.load("D:/data/PCB.npy")
    print(s.shape)
    #
    # s = s[0:128,:,:]
    # np.save("D:/kerry.npy", s)
    # print(s.shape)
    #





