import torch
from unittest import TestCase
from raygun.torch.networks.ResNet import *


class TestResnetGenerator2D(TestCase):
    def test_forward(self):
        gen = ResnetGenerator2D()

        # Generate a random input tensor
        input = torch.randn((1, 1, 256, 256))

        # Verify that the output has the expected shape
        output = gen(input)
        self.assertEqual(output.shape, (1, 64, 256, 256))

    def test_constructor(self):
        # Verify that we can create a generator with non-default settings
        gen = ResnetGenerator2D(input_nc=3, n_downsampling=3, n_blocks=4)

        # Verify that the generator has the expected properties
        self.assertEqual(gen.input_nc, 3)
        self.assertEqual(gen.n_downsampling, 3)
        self.assertEqual(gen.n_blocks, 4)
        self.assertEqual(len(gen.model), 26)


class TestResnetBlock2D(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.padding_type = "reflect"
        self.norm_layer = torch.nn.BatchNorm2d
        self.use_dropout = True
        self.use_bias = False
        self.activation = torch.nn.ReLU
        self.block = ResnetBlock2D(
            self.dim, self.padding_type, self.norm_layer,
            self.use_dropout, self.use_bias, self.activation
        )
        self.input_shape = (4, self.dim, 16, 16)
        self.input_tensor = torch.randn(self.input_shape)

    def test_output_shape(self):
        expected_output_shape = (4, self.dim, 16, 16)
        self.assertEqual(self.block(self.input_tensor).shape, expected_output_shape)

    def test_output_min_max(self):
        output = self.block(self.input_tensor)
        self.assertLessEqual(output.min().item(), -1)
        self.assertGreaterEqual(output.max().item(), 1)

    def test_crop(self):
        x = torch.randn((4, 16, 32, 32))
        shape = (16, 16)
        cropped_x = self.block.crop(x, shape)
        expected_shape = (4, 16, 16, 16)
        self.assertEqual(cropped_x.shape, expected_shape)

    def test_forward_valid_padding(self):
        # Check that forward pass works with valid padding
        self.block.padding_type = "valid"
        output = self.block(self.input_tensor)
        expected_shape = (4, self.dim, 14, 14)
        self.assertEqual(output.shape, expected_shape)

    def test_build_conv_block(self):
        conv_block = self.block.build_conv_block(
            self.dim, self.padding_type, self.norm_layer,
            self.use_dropout, self.use_bias, self.activation
        )
        self.assertEqual(len(conv_block), 9)
        self.assertIsInstance(conv_block, torch.nn.Sequential)


class TestResnetGenerator3D(unittest.TestCase):
    def test_output_shape(self):
        # Check that the output of the generator has the expected shape
        generator = ResnetGenerator3D()
        input_shape = (1, 1, 64, 64, 64)
        output = generator(torch.randn(input_shape))
        self.assertEqual(output.shape, input_shape)

    def test_zero_padding(self):
        # Check that the generator produces the same output when padding_type is 'zeros' and when it is 'valid'
        input_shape = (1, 1, 64, 64, 64)
        input = torch.randn(input_shape)
        generator_zeros = ResnetGenerator3D(padding_type='zeros')
        generator_valid = ResnetGenerator3D(padding_type='valid')
        output_zeros = generator_zeros(input)
        output_valid = generator_valid(input)
        self.assertTrue(torch.allclose(output_zeros, output_valid, rtol=1e-3, atol=1e-3))

    def test_add_noise(self):
        # Check that the generator produces different output when add_noise is True and when it is False
        input_shape = (1, 1, 64, 64, 64)
        input = torch.randn(input_shape)
        generator_no_noise = ResnetGenerator3D(add_noise=False)
        generator_with_noise = ResnetGenerator3D(add_noise=True)
        output_no_noise = generator_no_noise(input)
        output_with_noise = generator_with_noise(input)
        self.assertFalse(torch.allclose(output_no_noise, output_with_noise, rtol=1e-3, atol=1e-3))


class TestResnetBlock3D(unittest.TestCase):

    def test_valid_padding(self):
        block = ResnetBlock3D(dim=16, padding_type="valid", norm_layer=torch.nn.BatchNorm3d, use_dropout=False, use_bias=True, activation=torch.nn.ReLU)

        # input tensor with shape (batch_size, num_channels, depth, height, width)
        x = torch.randn(1, 16, 8, 32, 32)
        out = block(x)

        # output tensor should have shape (batch_size, num_channels, depth-2, height-2, width-2)
        self.assertEqual(out.shape, (1, 16, 6, 30, 30))

    def test_same_padding(self):
        block = ResnetBlock3D(dim=32, padding_type="same", norm_layer=torch.nn.InstanceNorm3d, use_dropout=True, use_bias=False, activation=torch.nn.LeakyReLU)

        # input tensor with shape (batch_size, num_channels, depth, height, width)
        x = torch.randn(2, 32, 16, 64, 64)
        out = block(x)

        # output tensor should have the same shape as input tensor
        self.assertEqual(out.shape, x.shape)

    def test_reflect_padding(self):
        block = ResnetBlock3D(dim=64, padding_type="reflect", norm_layer=torch.nn.GroupNorm, use_dropout=True, use_bias=True, activation=torch.nn.ELU)

        # input tensor with shape (batch_size, num_channels, depth, height, width)
        x = torch.randn(4, 64, 4, 16, 16)
        out = block(x)

        # output tensor should have the same shape as input tensor
        self.assertEqual(out.shape, x.shape)

    def test_replicate_padding(self):
        block = ResnetBlock3D(dim=128, padding_type="replicate", norm_layer=torch.nn.BatchNorm3d, use_dropout=False, use_bias=False, activation=torch.nn.Sigmoid)

        # input tensor with shape (batch_size, num_channels, depth, height, width)
        x = torch.randn(3, 128, 32, 128, 128)
        out = block(x)

        # output tensor should have the same shape as input tensor
        self.assertEqual(out.shape, x.shape)


class TestResNet(unittest.TestCase):

    def test_init_2d(self):
        # Test that the ResNet initializes correctly for 2D input
        resnet = ResNet(ndims=2, input_nc=3, output_nc=1, ngf=64)
        self.assertIsInstance(resnet, ResNet)
        self.assertEqual(resnet.ndims, 2)
        self.assertEqual(resnet.input_nc, 3)
        self.assertEqual(resnet.output_nc, 1)
        self.assertEqual(resnet.ngf, 64)

    def test_init_3d(self):
        # Test that the ResNet initializes correctly for 3D input
        resnet = ResNet(ndims=3, input_nc=1, output_nc=2, ngf=32)
        self.assertIsInstance(resnet, ResNet)
        self.assertEqual(resnet.ndims, 3)
        self.assertEqual(resnet.input_nc, 1)
        self.assertEqual(resnet.output_nc, 2)
        self.assertEqual(resnet.ngf, 32)

    def test_init_invalid_ndims(self):
        # Test that an error is raised if an invalid number of dimensions is passed
        with self.assertRaises(ValueError):
            ResNet(ndims=4, input_nc=3, output_nc=1, ngf=64)


if __name__ == '__main__':
    unittest.main()