from upscale_util import Upscaler, UpscaleParam


def test_enhance():
    args =  UpscaleParam()
    args.output = 'outputs'
    args.model_path = 'models/RealESRGAN_x4plus.pth'
    args.model_name = 'RealESRGAN_x4plus.pth'
    upscaler = Upscaler(args)

    upscaler.enhance("/Users/norkts/Downloads/facefusion/images/fa2adf804eec0b14a7f4c655384604ef.png")
