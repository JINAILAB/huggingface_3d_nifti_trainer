import torch
import datasets
import nibabel as nib
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    NormalizeIntensity,
    ScaleIntensity,
    ThresholdIntensity,
    # Intensity Transforms
    RandGaussianNoise,
    RandGaussianSmooth,
    RandShiftIntensity,
    RandScaleIntensity,
    RandAdjustContrast,
    RandGaussianSharpen,
    RandHistogramShift,
    # Geometric Transforms
    RandFlip,
    RandRotate,
    RandZoom,
    RandAffine,
    Rand3DElastic,
    RandGridDistortion,
    # Simulated transforms
    RandCoarseDropout,
    RandCoarseShuffle,
)

class ClassificationPresetTrain:
    """
    Training transform with MONAI-based data augmentation for 3D medical images.
    Includes both intensity and geometric augmentations.
    """
    def __init__(
        self,
        *,
        crop_size=(64, 64, 64),
        # Normalization parameters
        clip_min_max=None,  # tuple (min, max) or None
        use_normalize_intensity=True,  # if False, use ScaleIntensity
        normalize_nonzero=True,
        scale_minv=0.0,
        scale_maxv=1.0,
        # Intensity augmentation parameters
        gaussian_noise_prob=0.15,
        gaussian_noise_std=0.1,
        gaussian_smooth_prob=0.15,
        gaussian_smooth_sigma=(0.5, 1.0),
        shift_intensity_prob=0.15,
        shift_intensity_offset=0.1,
        scale_intensity_prob=0.15,
        scale_intensity_factors=(0.9, 1.1),
        adjust_contrast_prob=0.15,
        contrast_range=(0.75, 1.25),
        gaussian_sharpen_prob=0.1,
        sharpen_sigma=(0.5, 1.0),
        histogram_shift_prob=0.1,
        # Geometric augmentation parameters
        flip_prob=0.5,
        rotate_prob=0.2,
        rotate_range=0.174,  # ±10 degrees in radians
        zoom_prob=0.2,
        zoom_range=(0.9, 1.1),
        affine_prob=0.2,
        affine_rotate_range=0.174,
        affine_shear_range=0.1,
        affine_scale_range=(0.9, 1.1),
        grid_distortion_prob=0.1,
        grid_distortion_num_cells=5,
        grid_distortion_distort_limit=0.05,
        # Simulated low resolution
        coarse_dropout_prob=0.1,
        coarse_dropout_holes=8,
        coarse_dropout_spatial_size=(4, 4, 4),
    ): 
        transforms = [
            EnsureChannelFirst(channel_dim='no_channel'),
        ]
        
        # Intensity transforms
        if gaussian_noise_prob > 0:
            transforms.append(
                RandGaussianNoise(prob=gaussian_noise_prob, std=gaussian_noise_std)
            )
        
        if gaussian_smooth_prob > 0:
            transforms.append(
                RandGaussianSmooth(
                    prob=gaussian_smooth_prob,
                    sigma_x=gaussian_smooth_sigma,
                    sigma_y=gaussian_smooth_sigma,
                    sigma_z=gaussian_smooth_sigma,
                )
            )
        
        if shift_intensity_prob > 0:
            transforms.append(
                RandShiftIntensity(
                    offsets=shift_intensity_offset,
                    prob=shift_intensity_prob
                )
            )
        
        if scale_intensity_prob > 0:
            transforms.append(
                RandScaleIntensity(
                    factors=scale_intensity_factors,
                    prob=scale_intensity_prob
                )
            )
        
        if adjust_contrast_prob > 0:
            transforms.append(
                RandAdjustContrast(
                    prob=adjust_contrast_prob,
                    gamma=contrast_range
                )
            )
        
        if gaussian_sharpen_prob > 0:
            transforms.append(
                RandGaussianSharpen(
                    prob=gaussian_sharpen_prob,
                    sigma1_x=sharpen_sigma,
                    sigma1_y=sharpen_sigma,
                    sigma1_z=sharpen_sigma,
                    sigma2_x=sharpen_sigma,
                    sigma2_y=sharpen_sigma,
                    sigma2_z=sharpen_sigma,
                )
            )
        
        if histogram_shift_prob > 0:
            transforms.append(
                RandHistogramShift(prob=histogram_shift_prob)
            )
        
        # Geometric transforms
        if flip_prob > 0:
            transforms.extend([
                RandFlip(prob=flip_prob, spatial_axis=0),  # z-axis
                RandFlip(prob=flip_prob, spatial_axis=1),  # y-axis
                RandFlip(prob=flip_prob, spatial_axis=2),  # x-axis
            ])
        
        if rotate_prob > 0:
            transforms.append(
                RandRotate(
                    range_x=rotate_range,
                    range_y=rotate_range,
                    range_z=rotate_range,
                    prob=rotate_prob,
                    keep_size=True,
                )
            )
        
        if zoom_prob > 0:
            transforms.append(
                RandZoom(
                    prob=zoom_prob,
                    min_zoom=zoom_range[0],
                    max_zoom=zoom_range[1],
                    keep_size=True,
                )
            )
        
        if affine_prob > 0:
            transforms.append(
                RandAffine(
                    prob=affine_prob,
                    rotate_range=(affine_rotate_range, affine_rotate_range, affine_rotate_range),
                    shear_range=(affine_shear_range, affine_shear_range, affine_shear_range),
                    scale_range=(affine_scale_range, affine_scale_range, affine_scale_range),
                    mode='bilinear',
                    padding_mode='border',
                )
            )
        
        if grid_distortion_prob > 0:
            transforms.append(
                RandGridDistortion(
                    prob=grid_distortion_prob,
                    num_cells=grid_distortion_num_cells,
                    distort_limit=(-grid_distortion_distort_limit, grid_distortion_distort_limit),
                )
            )
        
        # Simulated low resolution transform
        if coarse_dropout_prob > 0:
            transforms.append(
                RandCoarseDropout(
                    holes=coarse_dropout_holes,
                    spatial_size=coarse_dropout_spatial_size,
                    prob=coarse_dropout_prob,
                    fill_value=0,
                )
            )
        
        # Resize
        transforms.append(Resize(spatial_size=crop_size, mode='trilinear'))
        
        # Clip intensity if specified
        if clip_min_max is not None:
            transforms.append(
                ThresholdIntensity(threshold=clip_min_max[0], above=False, cval=clip_min_max[0])
            )
            transforms.append(
                ThresholdIntensity(threshold=clip_min_max[1], above=True, cval=clip_min_max[1])
            )
        
        # Final normalization: use either NormalizeIntensity or ScaleIntensity
        if use_normalize_intensity:
            transforms.append(NormalizeIntensity(nonzero=normalize_nonzero))
        else:
            transforms.append(ScaleIntensity(minv=scale_minv, maxv=scale_maxv))

        self.transforms = Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
    
    
class ClassificationPresetEval:
    """
    Evaluation transform with minimal preprocessing for 3D medical images.
    Only applies resizing and normalization without augmentation.
    """
    def __init__(
        self,
        *,
        crop_size=(64, 64, 64),
        clip_min_max=None,
        use_normalize_intensity=True,
        normalize_nonzero=True,
        scale_minv=0.0,
        scale_maxv=1.0,
    ):
        transforms = [
            EnsureChannelFirst(channel_dim='no_channel'),
            Resize(spatial_size=crop_size, mode='trilinear'),
        ]
        
        # Clip intensity if specified
        if clip_min_max is not None:
            transforms.append(
                ThresholdIntensity(threshold=clip_min_max[0], above=False, cval=clip_min_max[0])
            )
            transforms.append(
                ThresholdIntensity(threshold=clip_min_max[1], above=True, cval=clip_min_max[1])
            )
        
        # Final normalization: use either NormalizeIntensity or ScaleIntensity
        if use_normalize_intensity:
            transforms.append(NormalizeIntensity(nonzero=normalize_nonzero))
        # else:
        #     transforms.append(ScaleIntensity(minv=scale_minv, maxv=scale_maxv))

        self.transforms = Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
    



    
def preprocess_train(batch, train_transforms):
    batch["pixel_values"] = [
        train_transforms(nifti.get_fdata()) for nifti in batch["nifti"]
    ]
    return batch


def preprocess_valid(batch, valid_transforms):
    """Apply train_transforms across a batch."""
    if "nifti" in batch:
        batch["pixel_values"] = [
            valid_transforms(nifti.get_fdata()) for nifti in batch["nifti"]
    ]
    return batch



def load_dataset(dataset, args):
    """Load and prepare training and validation datasets with transforms."""
    crop_size = getattr(args, "crop_size", (64, 64, 64))
    
    # Normalization parameters
    clip_min_max = getattr(args, "clip_min_max", None)
    use_normalize_intensity = getattr(args, "use_normalize_intensity", True)
    normalize_nonzero = getattr(args, "normalize_nonzero", False)
    scale_minv = getattr(args, "scale_minv", 0.0)
    scale_maxv = getattr(args, "scale_maxv", 1.0)
    
    # Intensity augmentation parameters
    gaussian_noise_prob = getattr(args, "gaussian_noise_prob", 0.15)
    gaussian_noise_std = getattr(args, "gaussian_noise_std", 0.1)
    gaussian_smooth_prob = getattr(args, "gaussian_smooth_prob", 0.15)
    gaussian_smooth_sigma = getattr(args, "gaussian_smooth_sigma", (0.5, 1.0))
    shift_intensity_prob = getattr(args, "shift_intensity_prob", 0.15)
    shift_intensity_offset = getattr(args, "shift_intensity_offset", 0.1)
    scale_intensity_prob = getattr(args, "scale_intensity_prob", 0.15)
    scale_intensity_factors = getattr(args, "scale_intensity_factors", (0.9, 1.1))
    adjust_contrast_prob = getattr(args, "adjust_contrast_prob", 0.15)
    contrast_range = getattr(args, "contrast_range", (0.75, 1.25))
    gaussian_sharpen_prob = getattr(args, "gaussian_sharpen_prob", 0.1)
    sharpen_sigma = getattr(args, "sharpen_sigma", (0.5, 1.0))
    histogram_shift_prob = getattr(args, "histogram_shift_prob", 0.1)
    
    # Geometric augmentation parameters
    flip_prob = getattr(args, "flip_prob", 0.5)
    rotate_prob = getattr(args, "rotate_prob", 0.2)
    rotate_range = getattr(args, "rotate_range", 0.174)
    zoom_prob = getattr(args, "zoom_prob", 0.2)
    zoom_range = getattr(args, "zoom_range", (0.9, 1.1))
    affine_prob = getattr(args, "affine_prob", 0.2)
    affine_rotate_range = getattr(args, "affine_rotate_range", 0.174)
    affine_shear_range = getattr(args, "affine_shear_range", 0.1)
    affine_scale_range = getattr(args, "affine_scale_range", (0.9, 1.1))
    grid_distortion_prob = getattr(args, "grid_distortion_prob", 0.1)
    grid_distortion_num_cells = getattr(args, "grid_distortion_num_cells", 5)
    grid_distortion_distort_limit = getattr(args, "grid_distortion_distort_limit", 0.05)
    
    # Simulated low resolution parameters
    coarse_dropout_prob = getattr(args, "coarse_dropout_prob", 0.1)
    coarse_dropout_holes = getattr(args, "coarse_dropout_holes", 8)
    coarse_dropout_spatial_size = getattr(args, "coarse_dropout_spatial_size", (4, 4, 4))
    
    train_transform = ClassificationPresetTrain(
        crop_size=crop_size,
        # Normalization
        clip_min_max=clip_min_max,
        use_normalize_intensity=use_normalize_intensity,
        scale_minv=scale_minv,
        scale_maxv=scale_maxv,
        # Intensity augmentation
        gaussian_noise_prob=gaussian_noise_prob,
        gaussian_noise_std=gaussian_noise_std,
        gaussian_smooth_prob=gaussian_smooth_prob,
        gaussian_smooth_sigma=gaussian_smooth_sigma,
        shift_intensity_prob=shift_intensity_prob,
        shift_intensity_offset=shift_intensity_offset,
        scale_intensity_prob=scale_intensity_prob,
        scale_intensity_factors=scale_intensity_factors,
        adjust_contrast_prob=adjust_contrast_prob,
        contrast_range=contrast_range,
        gaussian_sharpen_prob=gaussian_sharpen_prob,
        sharpen_sigma=sharpen_sigma,
        histogram_shift_prob=histogram_shift_prob,
        # Geometric augmentation
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
        rotate_range=rotate_range,
        zoom_prob=zoom_prob,
        zoom_range=zoom_range,
        affine_prob=affine_prob,
        affine_rotate_range=affine_rotate_range,
        affine_shear_range=affine_shear_range,
        affine_scale_range=affine_scale_range,
        grid_distortion_prob=grid_distortion_prob,
        grid_distortion_num_cells=grid_distortion_num_cells,
        grid_distortion_distort_limit=grid_distortion_distort_limit,
        # Simulated low resolution
        coarse_dropout_prob=coarse_dropout_prob,
        coarse_dropout_holes=coarse_dropout_holes,
        coarse_dropout_spatial_size=coarse_dropout_spatial_size,
    )
    
    valid_transform = ClassificationPresetEval(
        crop_size=crop_size,
        clip_min_max=clip_min_max,
        use_normalize_intensity=use_normalize_intensity,
        normalize_nonzero=normalize_nonzero,
        scale_minv=scale_minv,
        scale_maxv=scale_maxv,
    )
    
    train_ds = dataset['train']
    val_ds = dataset['valid']
    
    train_ds.set_transform(lambda batch: preprocess_train(batch, train_transform))
    val_ds.set_transform(lambda batch: preprocess_valid(batch, valid_transform))

    return train_ds, val_ds, train_transform, valid_transform


def load_test_dataset(dataset, args):
    """Load and prepare test dataset with transforms."""
    crop_size = getattr(args, "crop_size", (64, 64, 64))
    clip_min_max = getattr(args, "clip_min_max", None)
    use_normalize_intensity = getattr(args, "use_normalize_intensity", True)
    normalize_nonzero = getattr(args, "normalize_nonzero", True)
    scale_minv = getattr(args, "scale_minv", 0.0)
    scale_maxv = getattr(args, "scale_maxv", 1.0)
    
    valid_transform = ClassificationPresetEval(
        crop_size=crop_size,
        clip_min_max=clip_min_max,
        use_normalize_intensity=use_normalize_intensity,
        normalize_nonzero=normalize_nonzero,
        scale_minv=scale_minv,
        scale_maxv=scale_maxv,
    )
    
    val_ds = dataset['test']
    
    val_ds.set_transform(lambda batch: preprocess_valid(batch, valid_transform))

    return val_ds, valid_transform