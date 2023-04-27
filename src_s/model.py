from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decorder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs) #64
    #print(x.shape, skip_features.shape) # test : make sure that the res resolution of the x and skip features are same >> Comment after test :  yay !
    x = Concatenate()([x, skip_features]) 
    x = conv_block(x, num_filters)
    
    return x

def build_vgg16_unet(input_shape):
    inputs = Input(shape=input_shape)

    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output #512
    s2 = vgg16.get_layer("block2_conv2").output #256
    s3 = vgg16.get_layer("block3_conv3").output #128
    s4 = vgg16.get_layer("block4_conv3").output #64

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output #32
    
    """ Decoder """
    d1 = decorder_block(b1, s4, 512) #the input for the first decoder block would be the output of the bridge
    d2 = decorder_block(d1, s3, 256) #the input for the second decoder block would be the output of the first decoder block
    d3 = decorder_block(d2, s2, 128)
    d4 = decorder_block(d3, s1, 64)
    
    #print(d4.shape) # test : we assume that the shape of d4 would be 512,512,64 >> Comment after test : True yay !


    """ Output: Binary Segmentation """
    #This output is for binary segmentation
    # outputs = Conv2d(1, 1, padding='same', activation='sigmoid')(d4)
    # model = Model(inputs, outputs, name="VGG16_U-net")
    # return model    
    
    """ Output: Multi Segmentation """
    #This is for multi-class segmentation
    outputs = Conv2D(5, 1, padding='same', activation='softmax')(d4)
    model = Model(inputs, outputs, name="VGG16_U-net")
    return model    


if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_vgg16_unet(input_shape)
    model.summary()