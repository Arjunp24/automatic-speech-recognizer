from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
from keras.regularizers import l2

def simple_rnn_model(input_dim, output_dim=29):

    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
 
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    bn_rnn = BatchNormalization()(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):

    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization()(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):

    input_data = Input(name='the_input', shape=(None, input_dim))
    input_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='input_rnn'))(input_data)
    bn_layer = BatchNormalization(name='input_bn')(input_rnn)
    
    for i in range(recur_layers):
        rnn_layer_name = 'rnn_' + str(i)
        bn_layer_name = 'bn_' + str(i)
        
        rnn_layer = GRU(units, activation="relu",return_sequences=True, 
                 implementation=2, name=rnn_layer_name)(bn_layer)
        bn_layer = BatchNormalization(name=bn_layer_name)(rnn_layer)
       
    time_dense = TimeDistributed(Dense(output_dim))(bn_layer)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):

    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='bidir_rnn'), merge_mode='concat')(input_data)
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim=161,units=200,recur_layers=2, l2_lambda=0.001,output_dim=29):

    input_data = Input(name='the_input', shape=(None, input_dim))
    input_rnn = Bidirectional(GRU(units, activation='relu',
                            return_sequences=True,
                            implementation=2, 
                            name='input_rnn',
                            kernel_regularizer=l2(l2_lambda), 
                            recurrent_regularizer=l2(l2_lambda)))(input_data)
    bn_layer = BatchNormalization(name='input_bn')(input_rnn)
    for i in range(recur_layers):
        rnn_layer_name = 'rnn_' + str(i)
        bn_layer_name = 'bn_' + str(i)
        rnn_layer = GRU(units, activation="relu",return_sequences=True, 
                 implementation=2, name=rnn_layer_name,
                 kernel_regularizer=l2(l2_lambda), 
                 recurrent_regularizer=l2(l2_lambda))(bn_layer)
        bn_layer = BatchNormalization(name=bn_layer_name)(rnn_layer)
        
    time_dense = TimeDistributed(Dense(output_dim, kernel_regularizer=l2(l2_lambda)))(bn_layer)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model