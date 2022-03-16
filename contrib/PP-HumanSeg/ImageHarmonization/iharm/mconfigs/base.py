from iharm.model.base import DeepImageHarmonization

BMCONFIGS = {
    'dih256': {
        'model': DeepImageHarmonization,
        'params': {
            'depth': 7
        }
    },
    'improved_dih256': {
        'model': DeepImageHarmonization,
        'params': {
            'depth': 7,
            'batchnorm_from': 2,
            'image_fusion': True
        }
    },
    'improved_sedih256': {
        'model': DeepImageHarmonization,
        'params': {
            'depth': 7,
            'batchnorm_from': 2,
            'image_fusion': True,
            'attend_from': 5
        }
    },
    'dih512': {
        'model': DeepImageHarmonization,
        'params': {
            'depth': 8
        }
    },
    'improved_dih512': {
        'model': DeepImageHarmonization,
        'params': {
            'depth': 8,
            'batchnorm_from': 2,
            'image_fusion': True
        }
    },
    'improved_sedih512': {
        'model': DeepImageHarmonization,
        'params': {
            'depth': 8,
            'batchnorm_from': 2,
            'image_fusion': True,
            'attend_from': 6
        }
    },
}
