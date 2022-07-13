import numpy as np

from nnunet.preprocessing.preprocessing import PreprocessorFor2D, GenericPreprocessor, resample_patient
from nnunet.preprocessing.cropping import get_case_identifier_from_npz, ImageCropper
import nibabel as nib


def itk_meta_to_affine(itk_orig, itk_direct, itk_zooms):
    itk_orig, itk_direct, itk_zooms = np.array(itk_orig), np.array(itk_direct), np.array(itk_zooms)
    itk_direct = itk_direct.reshape(3, 3)
    itk_zooms = np.diag(itk_zooms)
    # ITK's reference space is in LPS, as in DICOM, thus the first 2 components must be inverted
    itk_orig[:2] *= -1
    itk_direct[:2] *= -1

    affine = np.eye(4)
    affine[:3, :3] = itk_direct.dot(itk_zooms)
    affine[:3, -1] = itk_orig
    return affine


def affine_after_crop(old_affine, mins):
    crop_affine = np.eye(4)
    crop_affine[:3, 3] = mins
    affine = np.dot(old_affine, crop_affine)
    return affine


def create_bounding_box_mask(shape, zooms, bbox_shape_mm=(90.0, 110.0, 130.0)):
    shape = np.asarray(shape)
    center = np.round(shape / 2).astype("int")
    length = np.ceil(np.asarray(bbox_shape_mm) / zooms / 2).astype("int")
    mins = (center - length).clip(0)
    maxs = (center + length).clip(max=shape - 1)
    return mins, maxs
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(i, j) for i, j in zip(mins, maxs + 1))] = True
    return mask


def crop_to_prostate(data, affine):
    mins, maxs = create_bounding_box_mask(data.shape, nib.Nifti1Image(np.array([[[0]]]), affine).header.get_zooms()[:3])
    data = data[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]
    return data, affine_after_crop(affine, mins)


def crop_data_seg(data, properties, seg=None):
    before_shape = data.shape
    orig_affine = itk_meta_to_affine(properties["itk_origin"], properties["itk_direction"], properties["itk_spacing"])
    canon_affine = nib.as_closest_canonical(nib.Nifti1Image(np.array([[[0]]]), orig_affine)).affine
    if not np.allclose(orig_affine, canon_affine):
        raise ValueError("The input data must be in RAS+.")
    assert data.shape[0] == 1 and seg.shape[0] == 1
    # Note the transpose, as the dims in crop_bbox are z, y, x
    default_crop_affine = affine_after_crop(orig_affine, np.array(properties["crop_bbox"])[:, 0].T)
    data, new_affine = crop_to_prostate(data[0].T, default_crop_affine)
    data = data.T[None, ...]
    assert np.allclose(new_affine[:3, :3], orig_affine[:3, :3])
    assert np.allclose(new_affine[:3, :3], default_crop_affine[:3, :3])
    # Overwrite itk_origin, which is the only one that is affected, as the cropping is not undone
    new_orig = new_affine[:3, -1]
    # Invert first the transposed axis, as the reference frame is LPS instead of RAS
    new_orig[:2] *= -1
    properties["itk_origin"] = new_orig
    # Set crop_bbox to None, to exclude returning from cropping, as the additional cropping here complicates things
    properties["crop_bbox"] = None
    # Set original shape to after default cropping (probably not used anywhere, since we defined crop_bbox to None)
    properties["original_size_of_raw_data"] = properties["size_after_cropping"]
    # Set size_after_cropping which original holds the default after cropping size to the size defined now
    properties["size_after_cropping"] = data[0].shape
    if seg is None:
        print(f"Before/after prostate cropping data shape: {before_shape}/{data.shape}")
        return data, properties
    seg_before_shape = seg.shape
    seg, new_affine = crop_to_prostate(seg[0].T, orig_affine)
    seg = seg.T[None, ...]
    print(
        "Before/after prostate cropping data and seg shape: "
        f"{before_shape}/{data.shape}, {seg_before_shape}/{seg.shape}"
    )
    return data, properties, seg


class ProstatePreprocessor(GenericPreprocessor):
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            "spacing": properties["original_spacing"],
            "spacing_transposed": original_spacing_transposed,
            "data.shape (data is transposed)": data.shape,
        }
        # remove nans.
        data[np.isnan(data)] = 0
        # Crop image to specific bounding box including mainly the prostate
        data, properties, seg = crop_data_seg(data, properties, seg=seg)
        data, seg = resample_patient(
            data,
            seg,
            np.array(original_spacing_transposed),
            target_spacing,
            self.resample_order_data,
            self.resample_order_seg,
            force_separate_z=force_separate_z,
            order_z_data=0,
            order_z_seg=0,
            separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold,
        )
        after = {"spacing": target_spacing, "data.shape (data is resampled)": data.shape}
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), (
            "self.normalization_scheme_per_modality " "must have as many entries as data has " "modalities"
        )
        assert len(self.use_nonzero_mask) == len(data), (
            "self.use_nonzero_mask must have as many entries as data" " has modalities"
        )

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]["mean"]
                std_intensity = self.intensityproperties[c]["sd"]
                lower_bound = self.intensityproperties[c]["percentile_00_5"]
                upper_bound = self.intensityproperties[c]["percentile_99_5"]
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]["percentile_00_5"]
                upper_bound = self.intensityproperties[c]["percentile_99_5"]
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "noNorm":
                print("no intensity normalization")
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                    data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                    data[c][mask == 0] = 0
                else:
                    mn = data[c].mean()
                    std = data[c].std()
                    # print(data[c].shape, data[c].dtype, mn, std)
                    data[c] = (data[c] - mn) / (std + 1e-8)
        return data, seg, properties

    def preprocess_test_case(self, data_files, target_spacing, seg_file=None, force_separate_z=None):
        data, seg, properties = ImageCropper.crop_from_list_of_files(data_files, seg_file)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(
            data, target_spacing, properties, seg, force_separate_z=force_separate_z
        )
        return data.astype(np.float32), seg, properties


class ProstatePreprocessorFor2D(PreprocessorFor2D):
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            "spacing": properties["original_spacing"],
            "spacing_transposed": original_spacing_transposed,
            "data.shape (data is transposed)": data.shape,
        }
        # remove nans.
        data[np.isnan(data)] = 0
        # Crop image to specific bounding box including mainly the prostate
        data, properties, seg = crop_data_seg(data, properties, seg=seg)

        data, seg = resample_patient(
            data,
            seg,
            np.array(original_spacing_transposed),
            target_spacing,
            self.resample_order_data,
            self.resample_order_seg,
            force_separate_z=force_separate_z,
            order_z_data=0,
            order_z_seg=0,
            separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold,
        )
        after = {"spacing": target_spacing, "data.shape (data is resampled)": data.shape}
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), (
            "self.normalization_scheme_per_modality " "must have as many entries as data has " "modalities"
        )
        assert len(self.use_nonzero_mask) == len(data), (
            "self.use_nonzero_mask must have as many entries as data" " has modalities"
        )

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]["mean"]
                std_intensity = self.intensityproperties[c]["sd"]
                lower_bound = self.intensityproperties[c]["percentile_00_5"]
                upper_bound = self.intensityproperties[c]["percentile_99_5"]
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]["percentile_00_5"]
                upper_bound = self.intensityproperties[c]["percentile_99_5"]
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "noNorm":
                print("no intensity normalization")
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                    data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                    data[c][mask == 0] = 0
                else:
                    mn = data[c].mean()
                    std = data[c].std()
                    # print(data[c].shape, data[c].dtype, mn, std)
                    data[c] = (data[c] - mn) / (std + 1e-8)
        return data, seg, properties

    def preprocess_test_case(self, data_files, target_spacing, seg_file=None, force_separate_z=None):
        data, seg, properties = ImageCropper.crop_from_list_of_files(data_files, seg_file)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(
            data, target_spacing, properties, seg, force_separate_z=force_separate_z
        )
        return data.astype(np.float32), seg, properties
