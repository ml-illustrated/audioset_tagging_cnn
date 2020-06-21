import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import torch
from torch import nn
import torch.nn.functional as F

import librosa
import onnx
import onnx_coreml
import coremltools

from torchlibrosa.stft import Spectrogram, LogmelFilterBank, STFT
from pytorch_utils import do_mixup, interpolate, pad_framewise_output

from models import MobileNetV1Framewise

# this works with all feature layers
class MobileNetV1Base(nn.Module):
    def __init__(self, *args, **kwargs):

        super(MobileNetV1Base, self).__init__(*args, **kwargs)

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(oup), 
                nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            '''
            init_layer(_layers[0])
            init_bn(_layers[2])
            '''
            return _layers

        def conv_dw(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(inp), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup), 
                nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            '''
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            '''
            return _layers

        '''
        self.features = nn.Sequential(
            conv_bn(  1,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
        )
        '''
        self.features = nn.Sequential(
            conv_bn(  1,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1))

        
    def forward(self, input):
        x = self.features(input)
        return x
        
class MobileNetV1FramewiseExport(MobileNetV1Framewise):
    def __init__(self, *args, **kwargs):
        
        super(MobileNetV1FramewiseExport, self).__init__(*args, **kwargs)
        self.interpolate_ratio = 32

        n_fft=1024
        hop_length=320
        win_length=1024
        window='hann'
        center=True
        pad_mode='reflect'


        """
        self.bn0 = nn.BatchNorm2d(64)
        '''
        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)
        '''

        '''
        self.spectrogram_extractor = STFT(
            n_fft=1024, hop_length=320, 
            win_length=1024, window='hann', center=True, pad_mode='reflect', 
            freeze_parameters=True)
        '''
        
        self.spectrogram_extractor = Spectrogram(
            n_fft=1024, hop_length=320, 
            win_length=1024, window='hann', center=True, pad_mode='reflect', 
            freeze_parameters=True)


        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, 
            n_mels=64, fmin=50, fmax=14000, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        """
        
    def forward(self, x):
        # x = x[:, None, :]   # (batch_size, channels_num, data_length)
        #x2 = self.conv_real(x)
        #x = F.pad(x, pad=(0, 0, 1,1))
        #x0 = F.relu(x)
        
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        # frames_num = x.shape[2]

        # print( 'x shape: ', x.shape ) # x shape:  torch.Size([1, 1, 701, 64])

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        
        x = self.features(x)

        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1)#!!!!, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1)#!!!!!!!, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)

        return x
        
    def forward_orig(self, x, mixup_lambda=None):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        # frames_num = x.shape[2]

        # print( 'x shape: ', x.shape ) # x shape:  torch.Size([1, 1, 701, 64])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.features(x)

        
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        # TEMP DISABLE framewise_output = pad_framewise_output(framewise_output, frames_num)

        return framewise_output


def export_model( checkpoint_path ):
    audio_path = 'R9_ZSCveAHg_7s.wav'
    frames_per_second = 100
    model_args = {'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000, 'classes_num': 527}
    Model = MobileNetV1FramewiseExport
    model = Model(**model_args)

    # Model = MobileNetV1Base
    # model = Model()

    # print( model )
    
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=model_args['sample_rate'], mono=True)
    
    # sample_input = torch.from_numpy( waveform[None, None, :32000])    # (1, audio_length)
    sample_input = torch.from_numpy( waveform[None, :32000])    # (1, audio_length)
    print( 'waveform: ', sample_input.shape ) # waveform:  torch.Size([1, 224000])

    # sample_input = torch.zeros([1, 1, 701, 64]) # for feeding features directly, full waveform
    # sample_input_1 = torch.zeros([1, 1, 101, 64]) # for feeding features directly, one sample waveform

    
    # Forward
    model.eval()
    batch_output = model(sample_input)
    print( 'out: ', batch_output[0].shape ) # out:  torch.Size([1, 1024, 3, 2]) torch.Size([1, 1, 101, 64])
    # sample_input_1 = batch_output[1]

    fn_onnx = '/tmp/PANN_test.onnx'
    # torch.onnx.export(model, (sample_input,sample_input_1), fn_onnx ) #, input_names=['input.1'], output_names=['333'])
    torch.onnx.export(model, sample_input, fn_onnx ) #, input_names=['input.1'], output_names=['333'])

    return fn_onnx

def convert_to_coreml( fn_mlmodel, filename_onnx, sample_input ):

        '''
        torch_output = self.gen_torch_output( sample_input )
        # print( 'torch_output: shape %s\nsample %s ' % ( torch_output.shape, torch_output[:, :, :3, :3] ) )
        print( 'torch_output: shape ', ( torch_output.shape ) ) # (1, 1, 28, 64)

        # first convert to ONNX
        filename_onnx = '/tmp/wave__melspec_model.onnx'
        model.convert_to_onnx( filename_onnx, sample_input )

        onnx_output = self.gen_onnx_output( filename_onnx, sample_input )
        '''

        
        # set up for Core ML export
        convert_params = dict(
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',
        )

        mlmodel = onnx_coreml.convert(
            model=filename_onnx,
            **convert_params, 
        )

        # print(mlmodel._spec.description)

        # assert mlmodel != None, 'CoreML Conversion failed'

        mlmodel.save( fn_mlmodel )

        # mlmodel = coremltools.models.MLModel( fn_mlmodel )
        # _ = coremltools.models.MLModel( mlmodel._spec )
    
if __name__ == '__main__':
    checkpoint_path = 'MobileNetV1_mAP=0.389.pth'
    filename_onnx = export_model( checkpoint_path )

    fn_mlmodel = '/tmp/PANN.mlmodel'
    convert_to_coreml( fn_mlmodel, filename_onnx, sample_input=None )

# python3 pytorch/export.py 'MobileNetV1_mAP=0.389.pth'
# xcrun coremlc compile /tmp/PANN.mlmodel  /tmp/mlc_output

'''
import coremltools
fn_mlmodel='/tmp/PANN.mlmodel'
# fn_mlmodel='/tmp/wave__melspec.mlmodel'

# mlmodel = coremltools.models.MLModel( fn_mlmodel )

spec = coremltools.utils.load_spec( fn_mlmodel )

builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)

builder.inspect_input_features()
builder.inspect_output_features()
builder.inspect_layers(last=-1)
builder.inspect_conv_channels("Conv_1")

for layer in builder.spec.neuralNetwork.layers:
  print( layer.name )

mlmodel = coremltools.models.MLModel( builder.spec )


# shaper = coremltools.models.NeuralNetworkShaper(spec)

from coremltools.libcoremlpython import _MLModelProxy
mlmodel = _MLModelProxy( fn_mlmodel, True )
'''
