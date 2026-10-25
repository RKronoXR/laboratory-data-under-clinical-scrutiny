"""Recorded MONAI image input transformation."""
def _monai_input(image_path):
    """Match old LoadImaged + EnsureChannelFirstd + ScaleIntensityd exactly."""
    import torch
    from monai.data import PILReader
    from monai.transforms import LoadImage, ScaleIntensity, EnsureType
    # Historical PILReader default is reverse_indexing=True. Make it explicit
    # and transpose the resulting prediction back to canonical row/column axes.
    loader = LoadImage(reader=PILReader(reverse_indexing=True), image_only=True, ensure_channel_first=True)
    image = loader(str(image_path))
    image = ScaleIntensity(minv=0.0,maxv=1.0)(image)
    return EnsureType(dtype=torch.float32)(image)
