import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import torch
from torch import nn
import torch.nn.functional as F

import librosa
import onnx
import onnx_coreml
import onnxruntime
import coremltools

from pytorch_utils import interpolate, pad_framewise_output

from models import MobileNetV1Framewise

        
class MobileNetV1FramewiseExport(MobileNetV1Framewise):
    def __init__(self, *args, **kwargs):
        
        super(MobileNetV1FramewiseExport, self).__init__(*args, **kwargs)
        self.interpolate_ratio = 32

        self.input_name = 'input.1'
        # self.output_names = ['196', '341' ] # looked up via Netron
        self.output_names = ['clip_output', 'frame_output', 'melspec' ]

    def forward_debug(self, x, mixup_lambda=None):
        
        #  (1, 1024, 3, 2)

        x = torch.mean(x, dim=3)
        # print( 'x mean: ', x.shape ) # x mean:  torch.Size([1, 1024, 3])

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)

        y=x1+x2
        
        '''
        x = x.unsqueeze(3)
        x1 = self.max_pool(x)
        x2 = self.avg_pool(x)
        x1 = x1.squeeze(3)
        x2 = x2.squeeze(3)
        '''

        return y, x1, x2
        
    def forward_test(self, x):
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
        
    def forward(self, x, mixup_lambda=None):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        melspec = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = melspec

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
        #  (1, 1024, 3, 2)

        x = torch.mean(x, dim=3)
        # print( 'x mean: ', x.shape ) # x mean:  torch.Size([1, 1024, 3])

        
        
        # original
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)


        '''
        # x = x.unsqueeze(-1)
        x = x[:,:,:,None]
        print( 'x_0 ', x.shape ) # torch.Size([1, 1024, 3, 1])
        p1d = (0, 0, 1, 1) # pad dim 2 by 1 on each side, dim 3 none
        x = F.pad(x, p1d, "constant", 0)  # effectively zero padding
        print( 'x_1 ', x.shape ) # torch.Size([1, 1024, 5, 1])
        x = x.squeeze(3)
        '''

        '''
        p1d = (1, 1) # pad dim 2 by 1 on each side, dim 3 none
        x = F.pad(x, p1d, "constant", 0)  # effectively zero padding
        features = x
        # print( 'padded: ', x.shape ) # torch.Size([1, 1024, 5])

        
        x1 = F.max_pool1d(x, kernel_size=3, stride=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1)
        # print( 'x1 x2: ', x1.shape, x2.shape ) # torch.Size([1, 1024, 3]) torch.Size([1, 1024, 3])
        '''

        x = x1 + x2
        features = x        
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        # TEMP DISABLE framewise_output = pad_framewise_output(framewise_output, frames_num)

        return clipwise_output, features, melspec

    def gen_torch_output( self, sample_input ):
        # Forward
        with torch.no_grad():
            raw_outputs = self( torch.from_numpy( sample_input ) )
            torch_outputs = [ item.cpu().detach().numpy() for item in raw_outputs ]
        
        for output in torch_outputs:
            print( 'out: ', output.shape )
        # out:  torch.Size([1, 96, 527])
        # out:  torch.Size([1, 1, 101, 64])

        return torch_outputs

    def convert_to_onnx( self, filename_onnx, sample_input ):

        input_names = [ self.input_name ]
        output_names = self.output_names
        
        torch.onnx.export(
            self,
            torch.from_numpy( sample_input ),
            filename_onnx,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
            # operator_export_type=OperatorExportTypes.ONNX
        )

    def gen_onnx_outputs( self, filename_onnx, sample_input ):
        import onnxruntime
        
        session = onnxruntime.InferenceSession( filename_onnx, None)
        
        input_name = session.get_inputs()[0].name
        # output_names = [ item.name for item in session.get_outputs() ]

        raw_results = session.run([], {input_name: sample_input})

        return raw_results[0]
        
    def convert_to_coreml( self, fn_mlmodel, sample_input ):

        torch_output = self.gen_torch_output( sample_input )

        # clipwise = torch_output[0]
        # for idx in torch.topk(torch.from_numpy(clipwise[0,:]), k=5).indices.squeeze(0).tolist():
        # print( '%s: %0.3f' % ( idx, clipwise[0,idx] ) )


        # first convert to ONNX
        filename_onnx = '/tmp/PANN_model.onnx'
        self.convert_to_onnx( filename_onnx, sample_input )

        onnx_outputs = self.gen_onnx_outputs( filename_onnx, sample_input )
        
        # set up for Core ML export
        convert_params = dict(
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',
            custom_conversion_functions={'Pad':_convert_pad},            
        )

        mlmodel = onnx_coreml.convert(
            model=filename_onnx,
            **convert_params, 
        )

        # print(mlmodel._spec.description)

        # assert mlmodel != None, 'CoreML Conversion failed'

        mlmodel.save( fn_mlmodel )

        return torch_output

        """
        model_inputs = {
            self.input_name : sample_input
        }
        # do forward pass
        mlmodel_outputs = mlmodel.predict(model_inputs, useCPUOnly=True)

        # fetch the spectrogram from output dictionary
        mlmodel_output =  mlmodel_outputs[ self.output_names[0] ]
        # print( 'mlmodel_output: shape %s \nsample %s ' % ( mlmodel_output.shape, mlmodel_output[:,:,:3, :3] ) )
        print( 'mlmodel_output: shape ', ( mlmodel_output.shape ) )
        
        # mlmodel = coremltools.models.MLModel( fn_mlmodel )
        # _ = coremltools.models.MLModel( mlmodel._spec )
        """


def _convert_pad(builder, node, graph, err):
    from onnx_coreml._operators import _convert_pad as _convert_pad_orig

    pads = node.attrs['pads']
    print( 'node.name: ', node.name, pads )
    
    if node.name != 'Pad_136':
        _convert_pad_orig( builder, node, graph, err )

    else:

        params_dict = {}
        params_dict['pad_l'] = pads[2]
        params_dict['pad_r'] = pads[5]
        params_dict['pad_t'] = 0
        params_dict['pad_b'] = 0
        params_dict['value'] = 0.0
        params_dict['mode']  = 'constant'
    

        builder.add_padding(
            name=node.name,
            left=params_dict['pad_l'],
            right=params_dict['pad_r'],
            top=params_dict['pad_t'],
            bottom=params_dict['pad_b'],
            value=params_dict['value'],
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            padding_type=params_dict['mode']
        )


        
def export_model( fn_mlmodel, fn_json, checkpoint_path ):
    # audio_path = 'R9_ZSCveAHg_7s.wav'
    audio_path = '/tmp/ring_hello.wav'

    frames_per_second = 100
    model_args = {'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000, 'classes_num': 527}
    Model = MobileNetV1FramewiseExport
    model = Model(**model_args)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load audio
    sample_rate = model_args['sample_rate']
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    
    sample_input = waveform[None, :sample_rate]    # (1, audio_length)
    print( 'waveform: ', sample_input.shape ) # waveform:  torch.Size([1, 224000])

    # sample_input = torch.zeros([1, 1, 701, 64]) # for feeding features directly, full waveform
    # sample_input_1 = torch.zeros([1, 1, 101, 64]) # for feeding features directly, one sample waveform

    
        
    # sample_input_1 = batch_output[1]

    #waveform:  torch.Size([1, 32000])
    #out:  torch.Size([32, 527])


    # fn_onnx = '/tmp/PANN_test.onnx'
    # torch.onnx.export(model, sample_input, fn_onnx ) #, input_names=['input.1'], output_names=['333'])

    model_outputs = model.convert_to_coreml( fn_mlmodel, sample_input )

    save_model_output_as_json( fn_json, model_outputs )

def save_model_output_as_json( fn_output, model_outputs ):
    import json
    output_data = [
        model_outputs[0][0,:].tolist(), # clipwise
        model_outputs[1][0,:].tolist(), # framewise
        model_outputs[2][0,0,:].tolist(), # melspec
    ]
    with open( fn_output, 'w' ) as fp:
        json.dump( output_data, fp )

    
def export_model_debug( fn_mlmodel, fn_json, checkpoint_path ):
    import numpy as np
    sample_input = np.zeros((1, 1024, 3, 2), dtype=np.float32)

    Model = MobileNetV1FramewiseExport
    model_args = {'sample_rate': 32000, 'window_size': 1024, 'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000, 'classes_num': 527}

    model = Model(**model_args)

    model_outputs = model.convert_to_coreml( fn_mlmodel, sample_input )
    
    
if __name__ == '__main__':
    # checkpoint_path = 'MobileNetV1_mAP=0.389.pth'
    import sys
    checkpoint_path = sys.argv[1]

    fn_mlmodel = '/tmp/PANN.mlmodel'
    fn_json = '/tmp/PANN_out.ring_hello.json'

    export_model( fn_mlmodel, fn_json, checkpoint_path )
    # export_model_debug( fn_mlmodel, fn_json, checkpoint_path )

# python3 pytorch/export.py 'MobileNetV1_mAP=0.389.pth'
# xcrun coremlc compile /tmp/PANN.mlmodel  /tmp/mlc_output

'''
import coremltools
fn_mlmodel='/tmp/PANN.mlmodel'

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


'''
import soundfile as sf

fn_wav = 'R9_ZSCveAHg_7s.wav'

waveform, samplerate = sf.read( fn_wav )
num_samples = 32000
sample_input = waveform[ num_samples*2:num_samples*3 ] # sec 2 to 3

sf.write( '/tmp/ring_hello.wav', sample_input, samplerate )
'''
