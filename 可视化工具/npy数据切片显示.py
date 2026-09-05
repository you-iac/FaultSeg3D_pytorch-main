import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from PIL import Image
import numpy as np
import os
from tqdm import tqdm


def select_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='Select .npy file',
        filetypes=[('NumPy files', '*.npy'), ('All files', '*.*')]
    )
    root.destroy()
    return file_path if file_path else None


def get_imshow_params(data: np.ndarray):
    # Scan bounded chunks: prediction volumes need neither a full finite-value
    # copy nor the old np.unique sort over hundreds of millions of voxels.
    dmin, dmax = float('inf'), float('-inf')
    is_binary = True
    for start in range(0, data.size, 1_000_000):
        chunk = data.flat[start:start + 1_000_000]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            dmin = min(dmin, float(finite.min()))
            dmax = max(dmax, float(finite.max()))
            if is_binary:
                is_binary = bool(np.all((finite == 0) | (finite == 1)))
    if dmin == float('inf'):
        raise ValueError('No valid values in data (all NaN/Inf).')

    # Binary mask: keep black/white display.
    if is_binary:
        return {'cmap': 'gray', 'vmin': 0.0, 'vmax': 1.0}

    # Probability map [0, 1]: use gamma enhancement for low-probability details.
    is_probability = (dmin >= -1e-6) and (dmax <= 1.0 + 1e-6)
    if is_probability:
        return {'cmap': 'turbo', 'norm': mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)}

    # General amplitude data: percentile stretch for robust contrast.
    # Preserve exact original percentiles rather than estimating from samples.
    finite = data[np.isfinite(data)]
    vmin, vmax = np.percentile(finite, [1, 99])
    if vmax <= vmin:
        if dmax > dmin:
            vmin, vmax = dmin, dmax
        else:
            vmin, vmax = dmin - 1e-6, dmax + 1e-6

    cmap = 'seismic' if (dmin < 0 and dmax > 0) else 'gray'
    return {'cmap': cmap, 'vmin': float(vmin), 'vmax': float(vmax)}


def save_slice_image(slice_data: np.ndarray, save_path: str, fig_width: float,
                     fig_height: float, imshow_params: dict, image_scale=10):
    """Save at the original 200-DPI figure resolution by default.

    Each worker owns its normalization/colormap and image. No pyplot figures or
    shared rendering state are used. image_scale enlarges with nearest neighbors.
    """
    norm = copy(imshow_params['norm']) if 'norm' in imshow_params else mcolors.Normalize(
        vmin=imshow_params['vmin'], vmax=imshow_params['vmax'])
    cmap = matplotlib.colormaps.get_cmap(imshow_params['cmap']).copy()
    rgba = cmap(norm(np.ma.masked_invalid(slice_data)), bytes=True)
    # Match the old opaque white figure background for NaN/Inf pixels.
    rgba[rgba[..., 3] == 0] = 255
    with Image.fromarray(rgba) as original:
        if image_scale == 1:
            original.save(save_path, compress_level=1, dpi=(200, 200))
        else:
            size = (original.width * image_scale, original.height * image_scale)
            if image_scale == 10:
                # Original figsize=max(2, axis_length/20), saved at 200 DPI.
                size = (int(fig_width * 200) if fig_width is not None else max(400, size[0]),
                        int(fig_height * 200) if fig_height is not None else max(400, size[1]))
            with original.resize(size, Image.Resampling.NEAREST) as enlarged:
                enlarged.save(save_path, compress_level=1, dpi=(200, 200))


def save_volume_slices(data_or_path, output_dir=None, step=20, workers=None, image_scale=10):
    """Export T/X/Y slices, using up to four PNG workers by default.

    workers=1 disables concurrency. The default image_scale=10 preserves the
    original output dimensions, including the 400-pixel minimum per axis.
    image_scale=1 explicitly selects native-resolution output.
    """
    for name, value in [('step', step), ('image_scale', image_scale)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if workers is None:
        workers = min(16, os.cpu_count() or 1)
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    if isinstance(data_or_path, (str, os.PathLike)):
        data = np.load(data_or_path, mmap_mode='r')
        if output_dir is None:
            output_dir = os.path.splitext(str(data_or_path))[0]
    else:
        data = np.asarray(data_or_path)
        if output_dir is None:
            raise ValueError('output_dir is required when data_or_path is an array.')

    if data.ndim != 3:
        raise ValueError(f'Expected 3D data, got shape {data.shape}.')
    if not data.size:
        raise ValueError('Expected non-empty 3D data.')

    output_dir = str(output_dir)
    print('shape:', data.shape)
    print('range:', float(np.nanmin(data)), '~', float(np.nanmax(data)))
    print('slice output:', output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Save one slice every N indices.
    imshow_params = get_imshow_params(data)

    def save_one(axis, index, prefix):
        selection = [slice(None)] * 3
        selection[axis] = index
        slice_data = data[tuple(selection)]  # A view, not a volume copy per worker.
        save_slice_image(slice_data, os.path.join(output_dir, f'{prefix}-{index}.png'),
                         None, None, imshow_params, image_scale=image_scale)

    jobs = [(axis, index, prefix) for axis, prefix in enumerate(('T', 'X', 'Y'))
            for index in range(0, data.shape[axis], step)]
    print(f'PNG workers: {workers}; image scale: {image_scale}x')
    with tqdm(total=len(jobs), desc='Saving slices') as pbar:
        if workers == 1:
            for job in jobs:
                save_one(*job)
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(save_one, *job) for job in jobs]
                for future in as_completed(futures):
                    future.result()  # Propagate write failures to the caller.
                    pbar.update(1)

    return output_dir


if __name__ == '__main__':
    selected_path = select_file()
    if selected_path is None:
        print('No file selected, exit.')
        raise SystemExit(0)

    save_volume_slices(selected_path)
