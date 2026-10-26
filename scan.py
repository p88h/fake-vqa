"""Make rendered pages look scanned: resolution, skew, blur, noise and JPEG artifacts.

The page is rendered at `dpi * oversample` and box-filtered down to `dpi` (the
scanner's sampling), then skewed (keeping the page size: the margins absorb the
corners), blurred by the optics, gets sensor noise and is
finally JPEG-compressed. Skew and blur work at the output resolution, which looks
the same as applying them before sampling but is several times faster. Noise is
generated once and each page takes a random window of it, so degradation stays
cheap at high resolutions.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# The page layout is A4 at 88 DPI (728 x 1036 pixels).
BASE_DPI = 88


@dataclass(frozen=True)
class ScanParams:
    dpi: int = BASE_DPI  # output resolution
    oversample: float = 1.0  # render at dpi * oversample, then downsample to dpi
    rotation: float = 0.0  # max skew in degrees, uniform in [-rotation, rotation]
    blur: float = 0.0  # Gaussian blur radius, in output pixels
    noise: float = 0.0  # Gaussian noise standard deviation, in 0-255 levels
    jpeg_quality: int = 75  # Pillow's default

    @property
    def render_scale(self) -> float:
        """Scale of the rendered page relative to the 88 DPI layout."""
        return self.dpi * self.oversample / BASE_DPI


SCAN_PRESETS = {
    "clean": ScanParams(),
    "light": ScanParams(oversample=2, rotation=0.5, blur=0.4, noise=6, jpeg_quality=60),
    "medium": ScanParams(oversample=2, rotation=1.5, blur=0.7, noise=12, jpeg_quality=40),
    "heavy": ScanParams(oversample=2, rotation=3.0, blur=1.0, noise=20, jpeg_quality=25),
}


def parse_scan(spec: str | None) -> ScanParams:
    """Parse `<preset>[,key=value...]` or `key=value,...`, e.g. `medium,dpi=150`."""
    params = SCAN_PRESETS["clean"]
    if not spec:
        return params
    types = {f.name: f.type for f in dataclasses.fields(ScanParams)}
    for item in spec.split(","):
        key, sep, value = item.partition("=")
        if not sep:
            if key not in SCAN_PRESETS:
                raise ValueError(f"Unknown scan preset {key!r}; choose from {', '.join(SCAN_PRESETS)}")
            params = SCAN_PRESETS[key]
        elif key in types:
            params = dataclasses.replace(params, **{key: types[key](value)})
        else:
            raise ValueError(f"Unknown scan parameter {key!r}; choose from {', '.join(types)}")
    return params


class Scanner:
    def __init__(self, params: ScanParams, seed: int | None = None):
        self.params = params
        # Own generator, so the document content doesn't depend on the scan settings
        self.rng = np.random.default_rng(seed)
        self._noise: np.ndarray | None = None

    def _noise_window(self, height: int, width: int) -> np.ndarray:
        if self._noise is None or self._noise.shape[0] < height or self._noise.shape[1] < width:
            shape = (int(height * 1.5), int(width * 1.5))
            noise = self.rng.normal(0, self.params.noise, shape)
            self._noise = np.rint(noise).clip(-255, 255).astype(np.int16)
        y = self.rng.integers(0, self._noise.shape[0] - height + 1)
        x = self.rng.integers(0, self._noise.shape[1] - width + 1)
        return self._noise[y : y + height, x : x + width]

    def scan(self, image: Image.Image) -> Image.Image:
        """Degrade a page rendered at `params.render_scale`; returns it at `params.dpi`."""
        p = self.params
        if p.oversample != 1:
            size = (round(image.width / p.oversample), round(image.height / p.oversample))
            image = image.resize(size, Image.Resampling.BOX)
        if p.rotation:
            angle = self.rng.uniform(-p.rotation, p.rotation)
            image = image.rotate(
                angle, resample=Image.Resampling.BILINEAR, fillcolor="white"
            )
        if p.blur:
            image = image.filter(ImageFilter.GaussianBlur(p.blur))
        if p.noise:
            pixels = np.asarray(image, dtype=np.int16) + self._noise_window(image.height, image.width)[..., None]
            image = Image.fromarray(pixels.clip(0, 255).astype(np.uint8))
        return image

    def save(self, image: Image.Image, path: str) -> None:
        self.scan(image).save(path, quality=self.params.jpeg_quality)
