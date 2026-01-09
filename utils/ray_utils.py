import numpy as np
import data_types
def generate_rays(self):
    """Generate rays for all the views."""
    pixel_center = 0.5 if self.use_pixel_centers else 0.0
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")

    pixels = np.stack((x, y, -np.ones_like(x)), axis=-1)
    inverse_intrisics = np.linalg.inv(self.intrinsic_matrix[Ellipsis, :3, :3])
    camera_dirs = (inverse_intrisics[None, None, :] @ pixels[Ellipsis, None])[Ellipsis, 0]

    directions = (self.camtoworlds[:, None, None, :3, :3]
                @ camera_dirs[None, Ellipsis, None])[Ellipsis, 0]
    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                            directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    self.rays = data_types.Rays(origins=origins, directions=viewdirs)
