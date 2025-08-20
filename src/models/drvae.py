import numpy as np
import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt
import random
# Set global seeds for reproducibility
# seed=2
# tf.set_random_seed(seed)
# random.seed(seed)
# np.random.seed(seed)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.losses import MeanSquaredError


tf.disable_v2_behavior()
class DRVAE:
    def __init__(self, inputsXPor3D=(1, 1, 1, 1), inputsXRtp3D=(1, 1, 1, 1), inputsXNtg3D=(1, 1, 1, 1), 
                 inputsXPermi3D=(1, 1, 1, 1), inputXfluidUncertainties=(1, 1), n_outputs=1):
        # Hyperparameters
        self.dropout = 0.5
        learning_rate = 0.0001
        self.reg_weights = 0.0001
        self.opt_method = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model = self.__build_model(inputsXPor3D, inputsXRtp3D, inputsXNtg3D, inputsXPermi3D, inputXfluidUncertainties, n_outputs)

    def encoderBlock(self, uncertainty):
        width = 2
        kernel_size = 3
        #3038 #3040 0.0001
        #3041        
        def conv_bn_relu(nb_filter, kernel_size, stride=None):
            def conv_func(x):
                x1 = tf.keras.layers.Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size),
                                            bias_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.L2(self.reg_weights))(x)
                x_bn = tf.keras.layers.BatchNormalization()(x1)
                x_relu = tf.keras.layers.ReLU()(x_bn)
                return x1, x_relu
            return conv_func
        # CONV3D
        encod1, encod1_relu = conv_bn_relu(width ** 3, kernel_size)(uncertainty)
        encod1_relu = tf.keras.layers.Dropout(self.dropout)(encod1_relu)
        encod2, encod2_relu = conv_bn_relu(width ** 3, kernel_size)(encod1_relu)
        # POOL3D
        encod3 = tf.keras.layers.MaxPooling3D(pool_size=3)(encod2_relu)
        encod3 = tf.keras.layers.Dropout(self.dropout)(encod3)
        # 2 CONV3D
        # encod3 = residual_block(encod3, width ** 3, (3, 3, 3), activation_fn)
        encod4, encod4_relu = conv_bn_relu(width ** 4, kernel_size)(encod3)
        encod4_relu = tf.keras.layers.Dropout(self.dropout)(encod4_relu)
        encod5, encod5_relu = conv_bn_relu(width ** 4, kernel_size)(encod4_relu)
        # POOL3D
        encod6 = tf.keras.layers.MaxPooling3D(pool_size=2)(encod5_relu)
        encod6 = tf.keras.layers.Dropout(self.dropout)(encod6)
        # 3 CONV3D
        encod7, encod7_relu = conv_bn_relu(width ** 5, kernel_size)(encod6)
        encod7_relu = tf.keras.layers.Dropout(self.dropout)(encod7_relu)
        encod8, encod8_relu = conv_bn_relu(width ** 5, kernel_size)(encod7_relu)
        # POOL3D
        encod9 = tf.keras.layers.MaxPooling3D(pool_size=2)(encod8_relu)
        encod9 = tf.keras.layers.Dropout(self.dropout)(encod9)
        # 4 CONV3D
        encod10, encod10_relu = conv_bn_relu(width ** 6, kernel_size)(encod9)
        encod10_relu = tf.keras.layers.Dropout(self.dropout)(encod10_relu)
        encod11, encod11_relu = conv_bn_relu(width ** 6, kernel_size)(encod10_relu)
        # POOL3D
        encod12 = tf.keras.layers.MaxPooling3D(pool_size=2)(encod11_relu)
        encod12 = tf.keras.layers.Dropout(self.dropout)(encod12)
        
        encod13 = tf.keras.layers.Flatten()(encod12)
        encod13 = tf.keras.layers.Dense(256,
                                        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                                        bias_regularizer=tf.keras.regularizers.L2(1e-4),
                                        activity_regularizer=tf.keras.regularizers.L2(1e-5))(encod13)
        
        return encod13, encod11, encod10, encod8, encod7, encod5, encod4, encod2, encod1

    def decoderBlock(self, latenSpace, encod11, encod10, encod8, encod7, encod5, encod4, encod2, encod1):        
        width = 2
        kernel_size = 3
        #3038 #3040 0.0001
        #3041
        
        decod = tf.keras.layers.Dropout(self.dropout)(latenSpace)
        decod = tf.keras.layers.Dense(1094,
                                      kernel_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      bias_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      activity_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      activation ='relu')(decod)
        
        decod = tf.keras.layers.Dense(256,
                                      kernel_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      bias_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      activity_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      activation ='relu')(decod)
        
        decod = tf.keras.layers.Dropout(self.dropout)(decod)
        decod = tf.keras.layers.Dense(1 * 2 * 3 * width ** 6,
                                      kernel_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      bias_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      activity_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                      activation ='relu')(decod)
        
        decod = tf.keras.layers.Reshape((1, 2, 3, width ** 6))(decod)
        decod = tf.keras.layers.Dropout(self.dropout)(decod)

        def dconv_bn_nolinear(nb_filter, kernel_size, x, encod, encod2):
            def _dconv_bn(x):
                add = tf.keras.layers.Add()
                x = tf.keras.layers.Conv3DTranspose(nb_filter, 
                                                    (kernel_size, kernel_size, kernel_size), 
                                                    padding='same',
                                                    bias_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                                    activity_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                                    kernel_regularizer=tf.keras.regularizers.L2(self.reg_weights))(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
                x = add([x, encod])
                x = tf.keras.layers.Conv3DTranspose(nb_filter, 
                                                    (kernel_size, kernel_size, kernel_size), padding='same',
                                                    bias_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                                    activity_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                                    kernel_regularizer=tf.keras.regularizers.L2(self.reg_weights))(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
                x = add([x, encod2])
                return x
            return _dconv_bn


        # 4 UPSAMPLING                                          
        decod = tf.keras.layers.UpSampling3D((2,2,2))(decod)#TimeDistributed
        # 4 3DCONV
        decod = dconv_bn_nolinear(width ** 6, kernel_size, decod, encod11, encod10)(decod)
        
        # 3 UPSAMPLING                                          
        decod = tf.keras.layers.UpSampling3D((2,2,2))(decod)
        # 3 3DCONV
        decod = dconv_bn_nolinear(width ** 5, kernel_size, decod, encod8, encod7)(decod)
        
        # 2 UPSAMPLING                                          
        decod = tf.keras.layers.UpSampling3D((2,2,2))(decod)
        # 2 3DCONV
        decod = dconv_bn_nolinear(width ** 4, kernel_size, decod, encod5, encod4)(decod)
        
        # 1 UPSAMPLING                                          
        decod = tf.keras.layers.UpSampling3D((3,3,3))(decod)
        # 0 3DCONV
        decod = dconv_bn_nolinear(width ** 3, kernel_size, decod, encod2, encod1)(decod)
                         
        decod = tf.keras.layers.Conv3DTranspose(filters=width ** 2, kernel_size=(3, 3, 3),
                                                bias_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                                activity_regularizer=tf.keras.regularizers.L2(self.reg_weights),
                                                padding='same', 
                                                kernel_regularizer=tf.keras.regularizers.L2(self.reg_weights))(decod)  
        # RECONSTRUCTION
        decod = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=(1, 1, 1), activation='linear')(decod)
               
        return decod
        
    def __build_model(self, inputsXPor3D, inputsXRtp3D, inputsXNtg3D, inputsXPermi3D, inputXfluidUncertainties, n_outputs): 
        print('build model')
        Por3D = tf.keras.Input(shape=inputsXPor3D, name='Por3D_input')
        Rtp3D = tf.keras.Input(shape=inputsXRtp3D, name='Rtp3D_input')
        Ntg3D = tf.keras.Input(shape=inputsXNtg3D, name='Ntg3D_input')
        Permi3D = tf.keras.Input(shape=inputsXPermi3D, name='Permi3D_input')
        fluidUncertainties = tf.keras.Input(shape=inputXfluidUncertainties, name='fluidUncertainties_input')
        
        # Encode Porosity
        e1, Por3D_encod11, Por3D_encod10, Por3D_encod8, Por3D_encod7, Por3D_encod5, Por3D_encod4, Por3D_encod2, Por3D_encod1 = self.encoderBlock(Por3D)
        
        # Encode Rock type
        e2, Rtp3D_encod11, Rtp3D_encod10, Rtp3D_encod8, Rtp3D_encod7, Rtp3D_encod5, Rtp3D_encod4, Rtp3D_encod2, Rtp3D_encod1 = self.encoderBlock(Rtp3D)
        
        # Encode NTG
        e3, Ntg3D_encod11, Ntg3D_encod10, Ntg3D_encod8, Ntg3D_encod7, Ntg3D_encod5, Ntg3D_encod4, Ntg3D_encod2, Ntg3D_encod1 = self.encoderBlock(Ntg3D)
        
        # Encode Permiability I
        e4, Permi3D_encod11, Permi3D_encod10, Permi3D_encod8, Permi3D_encod7, Permi3D_encod5, Permi3D_encod4, Permi3D_encod2, Permi3D_encod1 = self.encoderBlock(Permi3D)
        
        # Scalar properties
        x7 = tf.keras.layers.Reshape((fluidUncertainties.shape[1],))(fluidUncertainties)
        
        # Regression
        concat = tf.keras.layers.Concatenate()([e1, e2, e3, e4, x7])
        
        # VARIATIONAL AUTOENCODER
        self.z_mean = tf.keras.layers.Dense(n_outputs,activation = 'linear')(concat)
        self.z_log_var = tf.keras.layers.Dense(n_outputs,activation = 'linear')(concat)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], n_outputs), mean=0, stddev=1)
            return z_mean+tf.exp(z_log_var/2)*epsilon

        # Reparameterization
        z = tf.keras.layers.Lambda(sampling, output_shape=(n_outputs,))([self.z_mean, self.z_log_var])
        
        reconstructionPOR3D  = self.decoderBlock(z, Por3D_encod11, Por3D_encod10, Por3D_encod8, Por3D_encod7, Por3D_encod5, Por3D_encod4, Por3D_encod2, Por3D_encod1)
        reconstructionRTP3D  = self.decoderBlock(z, Rtp3D_encod11, Rtp3D_encod10, Rtp3D_encod8, Rtp3D_encod7, Rtp3D_encod5, Rtp3D_encod4, Rtp3D_encod2, Rtp3D_encod1)
        reconstructionNTG3D  = self.decoderBlock(z, Ntg3D_encod11, Ntg3D_encod10, Ntg3D_encod8, Ntg3D_encod7, Ntg3D_encod5, Ntg3D_encod4, Ntg3D_encod2, Ntg3D_encod1)
        reconstructionPERM3D = self.decoderBlock(z, Permi3D_encod11, Permi3D_encod10, Permi3D_encod8, Permi3D_encod7, Permi3D_encod5, Permi3D_encod4, Permi3D_encod2, Permi3D_encod1)
        
        vae = tf.keras.Model(
            [Por3D, Rtp3D, Ntg3D, Permi3D, fluidUncertainties], 
            [reconstructionPOR3D, reconstructionRTP3D, reconstructionNTG3D, reconstructionPERM3D]
        )         

        # Initialize the VGG16 model and feature extractor once
        def build_feature_extractor():
            vgg16 = VGG16(weights='imagenet', include_top=False)

            # Specify the layers from which we will extract features
            content_layers = ['block4_conv3', 'block5_conv3']
            style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3']

            # Create a new model that outputs feature maps from the specified layers
            content_outputs = [vgg16.get_layer(layer).output for layer in content_layers]
            style_outputs = [vgg16.get_layer(layer).output for layer in style_layers]

            # Create the feature extractor model
            feature_extractor = tf.keras.Model(inputs=vgg16.input, outputs=content_outputs + style_outputs)
            return feature_extractor

        # Initialize the feature extractor once
        feature_extractor = build_feature_extractor()

        # Function to preprocess the input to be compatible with VGG16 (convert grayscale to RGB)
        def preprocess_input_for_vgg16(x):
            x = tf.image.grayscale_to_rgb(x)  # Convert grayscale to RGB by duplicating the channel
            x = tf.keras.applications.vgg16.preprocess_input(x)  # Apply VGG16 preprocessing
            return x

        # Define the perceptual loss function
        def perceptual_loss(y_true, y_pred):            
            # Preprocess the images to be compatible with VGG16
            y_true = preprocess_input_for_vgg16(y_true)
            y_pred = preprocess_input_for_vgg16(y_pred)

            # Extract features from target and generated outputs
            target_features = feature_extractor(y_true)
            generated_features = feature_extractor(y_pred)
            # Initialize MeanSquaredError loss function
            mse_loss_fn = MeanSquaredError()

            # Compute the perceptual loss by comparing feature maps
            loss = 0
            for nro, (target_feature, generated_feature) in enumerate(zip(target_features, generated_features)):
                # Calculate MSE between the features
                loss += mse_loss_fn(target_feature, generated_feature)

            return loss
        
        def perceptual_fn(y_true, y_pred):
            Por3D_t, Rtp3D_t, Ntg3D_t, Permi3D_t = y_true[0], y_true[1], y_true[2], y_true[3]
            reconstructionPOR3D, reconstructionRTP3D, reconstructionNTG3D, reconstructionPERM3D = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
            
            # Compute perceptual loss for each input and reconstruction
            perceptual_Por3D_t_value = perceptual_loss(Por3D_t, reconstructionPOR3D)
            perceptual_Rtp3D_t_value = perceptual_loss(Rtp3D_t, reconstructionRTP3D)
            perceptual_Ntg3D_t_value = perceptual_loss(Ntg3D_t, reconstructionNTG3D)
            perceptual_Permi3D_t_value = perceptual_loss(Permi3D_t, reconstructionPERM3D)
            
            # Sum the perceptual loss values
            perceptual_value = perceptual_Por3D_t_value + perceptual_Rtp3D_t_value + perceptual_Ntg3D_t_value + perceptual_Permi3D_t_value
            # perceptual_value *= 24 * 48 * 72
            return perceptual_value#tf.reduce_mean(perceptual_value)

        def mse_fn(y_true, y_pred):
            Por3D_t, Rtp3D_t, Ntg3D_t, Permi3D_t = y_true[0], y_true[1], y_true[2], y_true[3]
            reconstructionPOR3D, reconstructionRTP3D, reconstructionNTG3D, reconstructionPERM3D = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
            
            # MSE for each input and reconstruction
            loss_Por3D_mse_value = tf.keras.losses.mse(tf.reshape(Por3D_t, [-1]), tf.reshape(reconstructionPOR3D, [-1]))
            loss_Rtp3D_mse_value = tf.keras.losses.mse(tf.reshape(Rtp3D_t, [-1]), tf.reshape(reconstructionRTP3D, [-1]))
            loss_Ntg3D_mse_value = tf.keras.losses.mse(tf.reshape(Ntg3D_t, [-1]), tf.reshape(reconstructionNTG3D, [-1]))
            loss_Permi3D_mse_value = tf.keras.losses.mse(tf.reshape(Permi3D_t, [-1]), tf.reshape(reconstructionPERM3D, [-1]))
            
            # Sum the MSE losses
            reconstruction_loss = loss_Por3D_mse_value + loss_Rtp3D_mse_value + loss_Ntg3D_mse_value + loss_Permi3D_mse_value 
            # reconstruction_loss *= 24 * 48 * 72
            return tf.reduce_mean(reconstruction_loss)

        def drvae_fn(y_true, y_pred):
            # MSE Loss
            mse_loss_value = mse_fn(y_true, y_pred)
            
            # KL Divergence Loss
            loss_KL = -0.5 * tf.reduce_mean(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=-1)
            # loss_KL *= 24 * 48 * 72
            loss_KL = tf.reduce_mean(loss_KL)
            
            return mse_loss_value + loss_KL

        def loss_fn(y_true, y_pred):
            # dimOriginal = 24 * 38 * 73
            Por3D, Rtp3D, Ntg3D, Permi3D = y_true[0], y_true[1], y_true[2], y_true[3]
            reconstructionPOR3D, reconstructionRTP3D, reconstructionNTG3D, reconstructionPERM3D = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
            loss_reconPor3D = tf.keras.losses.mse(tf.reshape(Por3D, [-1]), tf.reshape(reconstructionPOR3D, [-1]))
            loss_reconRtp3D = tf.keras.losses.mse(tf.reshape(Rtp3D, [-1]), tf.reshape(reconstructionRTP3D, [-1]))
            loss_reconNtg3D = tf.keras.losses.mse(tf.reshape(Ntg3D, [-1]), tf.reshape(reconstructionNTG3D, [-1]))
            loss_reconPermi3D = tf.keras.losses.mse(tf.reshape(Permi3D, [-1]), tf.reshape(reconstructionPERM3D, [-1]))
            reconstruction_loss = loss_reconPor3D + loss_reconRtp3D + loss_reconNtg3D + loss_reconPermi3D 
            reconstruction_loss *= 24 * 48 * 72
            loss_KL = -0.5 * tf.reduce_mean(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=-1)
            perceptual_loss_value = perceptual_fn(y_true, y_pred)
            # return tf.reduce_mean(reconstruction_loss)
            # perceptual_loss_value = perceptual_loss(y_true, y_pred, encod5, encod11) 
            return tf.reduce_mean(reconstruction_loss + loss_KL+ perceptual_loss_value)
        
        # def loss_fn(y_true, y_pred):
        #     # dimOriginal = 24 * 38 * 73
        #     Por3D_t, Rtp3D_t, Ntg3D_t, Permi3D_t = y_true[0], y_true[1], y_true[2], y_true[3]
        #     reconstructionPOR3D, reconstructionRTP3D, reconstructionNTG3D, reconstructionPERM3D = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
        #     loss_reconPor3D = tf.keras.losses.mse(tf.reshape(Por3D_t, [-1]), tf.reshape(reconstructionPOR3D, [-1]))
        #     loss_reconRtp3D = tf.keras.losses.mse(tf.reshape(Rtp3D_t, [-1]), tf.reshape(reconstructionRTP3D, [-1]))
        #     loss_reconNtg3D = tf.keras.losses.mse(tf.reshape(Ntg3D_t, [-1]), tf.reshape(reconstructionNTG3D, [-1]))
        #     loss_reconPermi3D = tf.keras.losses.mse(tf.reshape(Permi3D_t, [-1]), tf.reshape(reconstructionPERM3D, [-1]))
        #     reconstruction_loss = loss_reconPor3D + loss_reconRtp3D + loss_reconNtg3D + loss_reconPermi3D 
        #     reconstruction_loss *= 24 * 48 * 72
        #     loss_KL = -0.5 * tf.reduce_mean(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=-1)
            
        #     # Compute perceptual loss for each input and reconstruction
        #     perceptual_Por3D_t_value = perceptual_loss(Por3D_t, reconstructionPOR3D)
        #     perceptual_Rtp3D_t_value = perceptual_loss(Rtp3D_t, reconstructionRTP3D)
        #     perceptual_Ntg3D_t_value = perceptual_loss(Ntg3D_t, reconstructionNTG3D)
        #     perceptual_Permi3D_t_value = perceptual_loss(Permi3D_t, reconstructionPERM3D)
        #     perceptual_loss_value = perceptual_Por3D_t_value + perceptual_Rtp3D_t_value + perceptual_Ntg3D_t_value + perceptual_Permi3D_t_value
            
        #     return tf.reduce_mean(reconstruction_loss + loss_KL + perceptual_loss_value)
        
        # loss_fn, drvae_fn, mse_fn, perceptual_fn
        def correlation(x, y):    
            mx = tf.math.reduce_mean(x)
            my = tf.math.reduce_mean(y)
            xm, ym = x-mx, y-my
            r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
            r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
            return r_num / r_den
        
        vae.compile(loss=loss_fn,
                    metrics = [correlation, loss_fn],
                    optimizer = self.opt_method)
        vae.summary()

        return vae