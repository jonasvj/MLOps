import pytest
import torch
from src.models.model import ImageClassifier

class TestModel:

    @pytest.mark.parametrize('height', [28, 37])
    @pytest.mark.parametrize('width', [28, 53])
    @pytest.mark.parametrize('channels', [1, 3])
    @pytest.mark.parametrize('classes', [5, 10])
    def test_output_shape(self, height, width, channels, classes):
        model =  ImageClassifier(
            height=height, width=width, channels=channels, classes=classes)
        rand_input = torch.rand((64, channels, height, width))
        output = model(rand_input)

        assert output.shape == torch.Size([64, classes])

    @pytest.mark.parametrize(
        'input_shape',
        [(64, 3, 28, 28, 100),
         (64, 28),
        pytest.param((64, 28, 28), marks=pytest.mark.xfail),
        pytest.param((64, 1, 28, 28), marks=pytest.mark.xfail),
        pytest.param((64, 3, 28, 28), marks=pytest.mark.xfail)])
    def test_input_warn(self, input_shape):
        model = ImageClassifier()
        rand_wrong_input = torch.rand(input_shape)

        with pytest.raises(ValueError):
            model(rand_wrong_input)
