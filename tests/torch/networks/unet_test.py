import unittest
import torch
import math
import torch.nn as nn
import numpy as np
from raygun.torch.networks.UNet import *


class TestConvPass(unittest.TestCase):

    def test_ConvPass_output_shape(self):
        input_nc = 1
        output_nc = 1
        kernel_sizes = [[3, 3, 3]]
        activation = torch.nn.ReLU
        padding = "valid"
        residual = False
        padding_mode = "reflect"
        norm_layer = None
        input_shape = (1, 1, 10, 10, 10)
        x = torch.randn(input_shape)

        conv_pass = ConvPass(input_nc, output_nc, kernel_sizes, activation, padding, residual, padding_mode, norm_layer)
        output = conv_pass(x)
        expected_shape = (1, 1, 8, 8, 8)

        self.assertEqual(output.shape, expected_shape)

    def test_ConvPass_residual_output(self):
        input_nc = 1
        output_nc = 1
        kernel_sizes = [[3, 3, 3]]
        activation = torch.nn.ReLU
        padding = "valid"
        residual = True
        padding_mode = "reflect"
        norm_layer = None
        input_shape = (1, 1, 10, 10, 10)
        x = torch.randn(input_shape)

        conv_pass = ConvPass(input_nc, output_nc, kernel_sizes, activation, padding, residual, padding_mode, norm_layer)
        output = conv_pass(x)
        expected_shape = (1, 1, 8, 8, 8)
        self.assertEqual(output.shape, expected_shape)

    def test_ConvPass_no_residual_output(self):
        input_nc = 1
        output_nc = 1
        kernel_sizes = [[3, 3, 3]]
        activation = torch.nn.ReLU
        padding = "valid"
        residual = False
        padding_mode = "reflect"
        norm_layer = None
        input_shape = (1, 1, 10, 10, 10)
        x = torch.randn(input_shape)

        conv_pass = ConvPass(input_nc, output_nc, kernel_sizes, activation, padding, residual, padding_mode, norm_layer)
        output = conv_pass(x)
        expected_shape = (1, 1, 8, 8, 8)
        self.assertEqual(output.shape, expected_shape)


class ConvDownsampleTestCase(unittest.TestCase):
    def setUp(self):
        self.input_nc = 1
        self.output_nc = 1
        self.kernel_sizes = (3,3)
        self.downsample_factor =  2
        self.activation = nn.ReLU
        self.padding = "valid"
        self.padding_mode = "reflect"
        self.norm_layer = nn.BatchNorm2d
        self.model = ConvDownsample(
            self.input_nc,
            self.output_nc,
            self.kernel_sizes,
            self.downsample_factor,
            self.activation,
            self.padding,
            self.padding_mode,
            self.norm_layer,
        )

    def test_shape(self):
        x = torch.randn(1, self.input_nc, 32, 32)
        y = self.model(x)
        self.assertEqual(y.shape, (1, self.output_nc, 15, 15))

    def test_norm_layer(self):
        self.assertIsInstance(self.model.conv_pass[-2], self.norm_layer)

    def test_activation(self):
        self.assertIsInstance(self.model.conv_pass[-1], self.activation)

    def test_kernel_size(self):
        conv_layer = self.model.conv_pass[0]
        self.assertEqual(conv_layer.kernel_size, self.kernel_sizes)

    def test_stride(self):
        conv_layer = self.model.conv_pass[0]
        self.assertEqual(conv_layer.stride, (self.downsample_factor, self.downsample_factor))

    def test_padding(self):
        conv_layer = self.model.conv_pass[0]
        self.assertEqual(conv_layer.padding, self.padding)

    def test_padding_mode(self):
        conv_layer = self.model.conv_pass[0]
        self.assertEqual(conv_layer.padding_mode, self.padding_mode)


class TestMaxDownsample(unittest.TestCase):
    def test_downsample_2d(self):
        downsample_factor = [2, 2]
        flexible = True
        model = MaxDownsample(downsample_factor, flexible)
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        self.assertEqual(list(y.size()[2:]), [16, 16])
        
    def test_downsample_3d(self):
        downsample_factor = [2, 2, 2]
        flexible = False
        model = MaxDownsample(downsample_factor, flexible)
        x = torch.randn(1, 3, 32, 32, 32)
        y = model(x)
        self.assertEqual(list(y.size()[2:]), [16, 16, 16])
        
    def test_mismatch_error(self):
        downsample_factor = [2, 2]
        flexible = False
        model = MaxDownsample(downsample_factor, flexible)
        x = torch.randn(1, 3, 33, 33)
        with self.assertRaises(RuntimeError):
            y = model(x)

"""
class TestUpsample(unittest.TestCase):
    def test_upsample_mode_nearest(self):
        upsample = Upsample(scale_factor=[2, 2], mode="nearest")
        input_tensor = torch.tensor([[1, 2], [3, 4]]).float().unsqueeze(0).unsqueeze(0)
        expected_output_tensor = torch.tensor([[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]]).float()

        self.assertTrue(torch.allclose(upsample(input_tensor), expected_output_tensor, rtol=1e-5, atol=1e-5))

    def test_upsample_mode_linear(self):
        upsample = Upsample(scale_factor=[2, 2], mode="linear")
        input_tensor = torch.tensor([[1, 2], [3, 4]]).float().unsqueeze(0)
        expected_output_tensor = torch.tensor([[[1.0, 1.5, 2.0, 2.0],
                                                 [1.5, 2.0, 2.5, 2.5],
                                                 [3.0, 3.5, 4.0, 4.0],
                                                 [3.5, 4.0, 4.5, 4.5]]]).float()
        
        self.assertTrue(torch.allclose(upsample(torch.zeros_like(expected_output_tensor), input_tensor)[:,1,...], expected_output_tensor, rtol=1e-5, atol=1e-5))
        
    def test_upsample_mode_transposed_conv(self):
        upsample = Upsample(scale_factor=[2, 2], mode="transposed_conv", input_nc=1, output_nc=1)
        input_tensor = torch.tensor([[1, 2], [3, 4]]).float().unsqueeze(0).unsqueeze(0)
        expected_output_tensor = torch.tensor([[[[4.0, 4.0, 4.0, 4.0],
                                                 [4.0, 4.0, 4.0, 4.0],
                                                 [3.0, 3.0, 3.0, 3.0],
                                                 [3.0, 3.0, 3.0, 3.0]],
                                                [[2.0, 2.0, 2.0, 2.0],
                                                 [2.0, 2.0, 2.0, 2.0],
                                                 [1.0, 1.0, 1.0, 1.0],
                                                 [1.0, 1.0, 1.0, 1.0]]]]).float()

        self.assertTrue(torch.allclose(upsample(input_tensor), expected_output_tensor, rtol=1e-5, atol=1e-5))
"""

class TestUNet(unittest.TestCase):
    def test_init(self):
        input_nc = 3
        ngf = 16
        fmap_inc_factor = 2
        downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
        unet = UNet(input_nc, ngf, fmap_inc_factor, downsample_factors)

        self.assertEqual(unet.ndims, 3)
        self.assertEqual(unet.num_levels, 4)
        self.assertEqual(unet.num_heads, 1)
        self.assertEqual(unet.input_nc, 3)
        self.assertIsInstance(unet, torch.nn.Module)
    
    def test_forward(self):
        input_nc = 3
        ngf = 16
        fmap_inc_factor = 2
        downsample_factors = [(2, 2, 2), (2, 2, 2)]
        unet = UNet(input_nc, ngf, fmap_inc_factor, downsample_factors, padding_type="same")

        input_data = torch.randn(1, 3, 64, 64, 64)
        output = unet(input_data)

        self.assertEqual(output.shape, (1, ngf, 64, 64, 64))


if __name__ == "__main__":
    unittest.main()
