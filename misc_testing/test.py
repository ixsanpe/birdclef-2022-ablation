from archived_code.simple_cnn import * 
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    w_in = 4096
    conv_depth = 1
    w2i = Wav2Spec()
    
    input = torch.normal(0, 1, (1, w_in))
    img = w2i(input)

    plt.imshow(img.squeeze().numpy())
    plt.show()

    w_in = img.shape[-1]
    h_in = img.shape[-2]

    model = WavImgCNN(
        conv_depth=conv_depth, 
        kernel_sizes=[3]*conv_depth, 
        pool_kernel_sizes=[2]*conv_depth,
        strides=[1]*conv_depth, 
        w_in=w_in, 
        h_in=h_in, 
        n_channels=[1]*(conv_depth+1), 
        fc_dims=[10], 
        paddings=None
    )
    print(model(input))
