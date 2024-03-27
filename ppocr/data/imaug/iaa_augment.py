# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/iaa_augment.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import imgaug
import imgaug as ia
from imgaug import parameters as iap
import imgaug.augmenters as iaa
import random
import cv2
import gc
import os

class CustomDetAug(object):
    def __init__(self, debug=False, save_img_path=None, n_save_imgs=2000, **kwargs) -> None:
        self.augmentation_pipeline = iaa.Sequential([
                    # # Augment color and brightness
                    iaa.Sometimes(0.7,
                        iaa.OneOf([
                            iaa.Add((-30, 50)),
                            iaa.AddToHue((-100, 100)),
                            iaa.AddToBrightness((-30, 30)),
                            iaa.AddToSaturation((-50, 50)),
                            iaa.AddToHueAndSaturation((-40, 40)),
                            iaa.ChangeColorTemperature((4000, 11000)),
                            iaa.Grayscale(alpha=(0.0, 1.0)),
                            # iaa.ChannelShuffle(1),
                            # iaa.Invert(0.1, per_channel=True),
                            iaa.BlendAlphaHorizontalLinearGradient(iaa.Add(iap.Normal(iap.Choice([-30, 30]), 20)), start_at=(0, 0.25), end_at=(0.75, 1)),
                            iaa.BlendAlphaHorizontalLinearGradient(iaa.Add(iap.Normal(iap.Choice([-30, 30]), 20)), start_at=(0.75, 1), end_at=(0, 0.25)),
                            iaa.MultiplyBrightness((0.75, 1.25)),
                            iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.25), add=(-20, 20)),
                            iaa.Multiply((0.85, 1.10)),
                            # Change contrast
                            iaa.SigmoidContrast(gain= (3, 7), cutoff=(0.3, 0.6)),
                            iaa.LinearContrast((0.7, 1.3)),
                            iaa.GammaContrast((0.7, 1.5)),
                            iaa.LogContrast(gain=(0.7, 1.3)),
                            iaa.pillike.Autocontrast((2, 5)),
                            iaa.Emboss(alpha=(0.1, 0.5), strength=(0.8, 1.2)),
                            ]),
                        ), 
                    # # # Noise and change background
                    iaa.Sometimes(0.4,
                        iaa.OneOf([
                            iaa.pillike.FilterSmoothMore(),
                            iaa.imgcorruptlike.Spatter(severity=(1,3)),
                            iaa.pillike.EnhanceSharpness(),
                            iaa.AdditiveGaussianNoise(scale=(0.02 * 255, 0.05 * 255)),
                            iaa.AdditiveGaussianNoise(scale=(0.02 * 255, 0.05 * 255), per_channel=True),
                            iaa.SaltAndPepper(p=(0.001, 0.01)),
                            iaa.Sharpen(alpha=(0.1, 0.5)),
                            iaa.MultiplyElementwise((0.9, 1.1), per_channel=0.5),
                            iaa.GaussianBlur(sigma=(0.5, 2)),
                            iaa.AverageBlur(k=(3, 7)),
                            iaa.MotionBlur(k=(3, 9), angle=(-180, 180)),
                            iaa.Dropout((0.001, 0.01), per_channel=True),
                            iaa.ElasticTransformation(alpha=(1, 10), sigma=(2, 4)),
                            iaa.CoarseDropout(0.02, size_percent=(0.01, 0.3), per_channel=True),
                            ])
                        ),
                    # # # Transform
                    # iaa.Sometimes(0.25,
                    #     iaa.OneOf([
                    #         iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                    #         iaa.Rotate((-2, 2)),
                    #         iaa.Rotate((-3, 3), fit_output=True, cval=(0,255), mode=ia.ALL),
                    #         iaa.Crop((1,3)),
                    #         iaa.ShearX((-5, 5), mode=ia.ALL),
                    #         iaa.ShearX((-9, 9), mode=ia.ALL, fit_output=True),
                    #         iaa.ShearY((-2, 2), mode=ia.ALL, cval=(0, 255)),
                    #         iaa.ShearY((-4, 4), mode=ia.ALL, cval=(0, 255), fit_output=True),
                    #         iaa.Affine(translate_px=(-2,4), mode=ia.ALL),
                    #         iaa.Affine(translate_px=(-4,4), mode=ia.ALL, fit_output=True),
                    #     ])
                    # ),
                    
                    # # compress image
                    iaa.Sometimes(0.03,
                        iaa.OneOf([
                            iaa.JpegCompression(compression=(50, 80)),
                            iaa.imgcorruptlike.Pixelate(severity=(1)),
                            iaa.UniformColorQuantization((10, 120)),
                        ])
                    )
                ])
        self.debug = True
        self.save_img_path = save_img_path
        self.n_save_imgs = n_save_imgs
        self.save_img_count = 0
        self.gc_count = 0
        if self.debug:
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            assert save_img_path != None, "Use debug must pass the save_img_path parameter"

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape
        if random.random() <= 0.65:
            img = self.augmentation_pipeline.augment(image=img.astype(np.uint8))
        if self.debug and self.save_img_count < self.n_save_imgs:
            cv2.imwrite(os.path.join(self.save_img_path, data['img_path'].split('/')[-1]), img.astype(np.uint8))
            self.save_img_count += 1
        data['image'] = img
        self.gc_count += 1
        if self.gc_count % 10000 == 0:
            gc.collect()
            self.gc_count = 0
        return data

class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(
                    *[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{
                k: self.to_tuple_if_list(v)
                for k, v in args['args'].items()
            })
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment():
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            augmenter_args = [{
                'type': 'Fliplr',
                'args': {
                    'p': 0.5
                }
            }, {
                'type': 'Affine',
                'args': {
                    'rotate': [-10, 10]
                }
            }, {
                'type': 'Resize',
                'args': {
                    'size': [0.5, 3]
                }
            }]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        image = data['image']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(
                keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
