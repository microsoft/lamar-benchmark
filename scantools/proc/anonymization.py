import logging
from pathlib import Path
from typing import List, Tuple, Optional
import tarfile
import cv2
import numpy as np

from ..utils.io import read_image, write_image

logger = logging.getLogger(__name__)

try:
    from redact.settings import Settings
    import redact.v3 as redact
except ImportError as e:
    logger.error('Could not import Brighter.AI API - did you install it?\n'
                 'pip install git+https://github.com/brighter-ai/redact-client.git')
    redact = Settings = None


def blur_detections(image: np.ndarray,
                    faces: List[Tuple[float]],
                    blend_ksize_multiplier: float = 0.2) -> Tuple[np.ndarray]:
    hi, wi = image.shape[:2]
    mask = np.full((hi, wi), 0.0)
    if len(faces) == 0:
        return image, mask

    blurred = image.copy()
    tmp = np.zeros_like(image)
    for bbox in faces:
        x1, y1 = np.floor(bbox[:2]).astype(int)
        x2, y2 = np.ceil(bbox[2:]).astype(int) + 1
        raw_w, raw_h = x2 - x1, y2 - y1
        blend_ksize = int(np.ceil(blend_ksize_multiplier * max(raw_w, raw_h)))
        if blend_ksize % 2 == 1:
            blend_ksize += 1

        # the box corners are sometimes out of the image boundaries
        w = raw_w + 3 * blend_ksize
        h = raw_h + 3 * blend_ksize
        padding = blend_ksize + blend_ksize // 2
        x1, y1 = x1 - padding, y1 - padding
        x2, y2 = x2 + padding, y2 + padding
        x1_, y1_ = max(x1, 0), max(y1, 0)
        x2_, y2_ = min(x2, wi), min(y2, hi)
        patch = np.s_[y1_:y2_, x1_:x2_]

        # define the mask as an ellipse in the bbox
        # extend ellipse to compensate for post-blurring
        dx = ((np.arange(x2 - x1) * 2 - w) / (raw_w + blend_ksize // 2)) ** 2
        dy = ((np.arange(y2 - y1) * 2 - h) / (raw_h + blend_ksize // 2)) ** 2
        mask_box = (dx[None] + dy[:, None]) <= 1
        if blend_ksize > 0:
            mask_box = cv2.blur(mask_box.astype(float), (blend_ksize,) * 2)
        # Crop the mask such that it is still centered.
        mask_box = mask_box[y1_ - y1 : y2_ - y1, x1_ - x1 : x2_ - x1]
        mask[patch] = np.clip(mask[patch] + mask_box, 0, 1)

        # blur a patch bigger than the bbox
        # the kernel size depends on the original bbox size to avoid artifacts
        ksize = max(raw_w, raw_h) // 2 + 1
        blur = cv2.blur(image[patch], (ksize,)*2)

        # copy the masked blur
        # hacky since the mask is over the bbox but the blur is over a larger patch
        tmp[patch] = blur
        if len(image.shape) == 3:
            mask_box = mask_box[..., None]
        blurred[patch] = mask_box * tmp[patch] + (1 - mask_box) * blurred[patch]
        # np.where(mask_box, tmp[patch], blurred[patch])

    blurred = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)

    return blurred, mask


class BrighterAIAnonymizer:
    labels_filename = 'labels.json'

    def __init__(self, apikey, **kwargs):
        self.apikey = apikey
        self.url = Settings().redact_online_url
        self.args = redact.JobArguments(**kwargs)

    def query_frame_labels(self, paths, tmp_dir):
        labels_path = tmp_dir / self.labels_filename
        if labels_path.exists():
            logger.info('Reading labels %s', labels_path)
            with open(labels_path, 'r') as fid:
                labels = redact.JobLabels.parse_raw(fid.read())
            return labels
        logger.info('Calling API with %d images in %s.', len(paths), tmp_dir)

        tmp_dir.mkdir(exist_ok=True, parents=True)
        tar_path = tmp_dir / f'{tmp_dir.name}.tar'
        with tarfile.open(tar_path, 'w') as tar:
            for i, p in enumerate(paths):
                tar.add(p, arcname=str(i) + p.suffix)

        # call the anonymization API
        instance = redact.RedactInstance.create(
            service='blur', out_type='archives', redact_url=self.url, api_key=self.apikey)
        with open(tar_path, 'rb') as fid:
            job = instance.start_job(file=fid, job_args=self.args)
            job.wait_until_finished()
        status = job.get_status()
        if status.state == redact.JobState.failed:
            logger.warning('Job failed for %s: %s, %s', tmp_dir.name, status.warnings, status.error)
            return None
        labels = job.get_labels()

        with open(labels_path, 'w') as fid:
            fid.write(labels.json())
        tar_path.unlink()
        return labels

    def face_is_valid(self, face, image_shape):
        if not self.args.single_frame_optimized:
            if face.score is None:
                logger.warning("Found face with score=None.")
                return True
            return face.score >= 0.5
        mx, my, Mx, My = face.bounding_box
        area_ratio = (
            (Mx - mx + 1) * (My - my + 1) / (image_shape[0] * image_shape[1]))
        # We use a conservative detection threshold of 40%. Regardless of the 
        # threshold, we noticed a significant number of false positives, notably
        # around reflections / bright regions in HoloLens images. These
        # detections cover a significant part of the image, and, given that most
        # of our data is captured at least a few meters away from bystanders, we
        # filter out all detection convering >=4% of total area.
        return face.score >= 0.40 and area_ratio < 0.04

    def blur_image_group(self, input_paths: List[Path], tmp_dir: Path,
                         output_paths: Optional[List[Path]] = None):
        labels = self.query_frame_labels(input_paths, tmp_dir)
        if labels is None:
            return None
        inplace = output_paths is None

        assert len(input_paths) == len(labels.frames), (tmp_dir, len(input_paths))
        counts = {'faces': 0, 'plates': 0}
        # actually blur the images
        for idx, image_path in enumerate(input_paths):
            label = labels.frames[idx]
            assert label.index == (idx+1)
            if not label.faces and not label.license_plates and inplace:
                continue

            image = read_image(image_path)
            faces = [f for f in label.faces if self.face_is_valid(f, image.shape)]
            plates = [f for f in label.license_plates if f.score >= 0.6]

            blurred, _ = blur_detections(image, [f.bounding_box for f in faces])
            blurred, _ = blur_detections(
                blurred, [f.bounding_box for f in plates], blend_ksize_multiplier=0)

            assert blurred.shape == image.shape and blurred.dtype == image.dtype
            counts['faces'] += len(faces)
            counts['plates'] += len(plates)

            if inplace:
                if len(faces) > 0 or len(plates) > 0:
                    write_image(image_path, blurred)
            else:
                out = output_paths[idx]
                out.parent.mkdir(exist_ok=True, parents=True)
                write_image(out, blurred)
        logger.info('Finished anonymization in %s', tmp_dir)
        return counts
