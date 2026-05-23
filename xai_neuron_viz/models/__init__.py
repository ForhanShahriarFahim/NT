# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2022- 09- 22
Authors: Yu wenlong  and  DRAGON_501

*************************************Import***********************************"""
import torch

# ************************************************************************************************


# Note: You need to change the ckpt_path in the following function to customize your own CoE case.
def build_models(args, model_name=None, resume=None):

    msg = None
    if model_name is not None:
        pass
    else:
        model_name = args.model_name

    if resume is not None:
        pass
    else:
        resume = args.resume

    if model_name == 'rn50':
        from torchvision.models.resnet import resnet50
        model = resnet50(pretrained=True)

        # Load your own trained resnet
        if resume is not None:
            assert resume.endswith('.pth'), 'The resume file must be a .pth file.'

            model_ckpt = torch.load(resume, map_location='cpu')

            new_state_dict = {}
            for k, v in model_ckpt['state_dict'].items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
            print('load ckpt from {}'.format(resume))
            print('loaded ckpt msg: {}'.format(msg))

        criterion = None
        preprocess = None
        tokenizer = None

    elif model_name == 'rn152' and 'imagenet' in args.dataset:
        from torchvision.models.resnet import resnet152
        model = resnet152(pretrained=True)
        criterion = None
        preprocess = None
        tokenizer = None
        
    # --- NT LAB PATCH: Added support for standard timm Vision Transformers ---
    elif model_name == 'vit_base_patch16_224' or model_name == 'vit':
        import timm
        print("Loading standard timm ViT-Base-16...")
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        criterion = None
        preprocess = None
        tokenizer = None
    # ---------------------------------------------------------------------------

    # --- NT LAB PATCH: Added support for standard torchvision VGG16 ---
    elif model_name == 'vgg16':
        from torchvision.models import vgg16
        print("Loading standard torchvision VGG16...")
        model = vgg16(pretrained=True)
        criterion = None
        preprocess = None
        tokenizer = None
    # ------------------------------------------------------------------

    else:
        print(f"ERROR: Model {model_name} not recognized by the gatekeeper!")
        return None, None, None, None, None

    return model, criterion, preprocess, tokenizer, msg