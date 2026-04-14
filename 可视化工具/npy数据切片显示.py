import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from tqdm import tqdm


def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='Select .npy file',
        filetypes=[('NumPy files', '*.npy'), ('All files', '*.*')]
    )
    root.destroy()
    return file_path if file_path else None


def get_imshow_params(data: np.ndarray):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError('No valid values in data (all NaN/Inf).')

    dmin = float(finite.min())
    dmax = float(finite.max())

    # Binary mask: keep black/white display.
    unique_vals = np.unique(finite)
    is_binary = unique_vals.size <= 2 and np.all(np.isin(unique_vals, [0.0, 1.0]))
    if is_binary:
        return {'cmap': 'gray', 'vmin': 0.0, 'vmax': 1.0}

    # Probability map [0, 1]: use gamma enhancement for low-probability details.
    is_probability = (dmin >= -1e-6) and (dmax <= 1.0 + 1e-6)
    if is_probability:
        return {'cmap': 'turbo', 'norm': mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)}

    # General amplitude data: percentile stretch for robust contrast.
    vmin, vmax = np.percentile(finite, [1, 99])
    if vmax <= vmin:
        if dmax > dmin:
            vmin, vmax = dmin, dmax
        else:
            vmin, vmax = dmin - 1e-6, dmax + 1e-6

    cmap = 'seismic' if (dmin < 0 and dmax > 0) else 'gray'
    return {'cmap': cmap, 'vmin': float(vmin), 'vmax': float(vmax)}


def save_slice_image(slice_data: np.ndarray, save_path: str, fig_width: float, fig_height: float, imshow_params: dict):
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(slice_data, aspect=1, interpolation='nearest', **imshow_params)
    plt.axis('off')
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    selected_path = select_file()
    if selected_path is None:
        print('No file selected, exit.')
        raise SystemExit(0)

    data = np.load(selected_path)
    filename = os.path.splitext(selected_path)[0]

    print('shape:', data.shape)
    print('range:', float(np.nanmin(data)), '~', float(np.nanmax(data)))

    os.makedirs(filename, exist_ok=True)

    # Save one slice every N indices.
    step = 20
    imshow_params = get_imshow_params(data)

    # dim0 -> T-i
    dim0_indices = list(range(0, data.shape[0], step))
    with tqdm(total=len(dim0_indices), desc='Processing dim0') as pbar:
        fig_width = max(2.0, data.shape[2] / 20.0)
        fig_height = max(2.0, data.shape[1] / 20.0)
        for i in dim0_indices:
            save_slice_image(data[i, :, :], f'{filename}/T-{i}.png', fig_width, fig_height, imshow_params)
            pbar.update(1)

    # dim1 -> X-i
    dim1_indices = list(range(0, data.shape[1], step))
    with tqdm(total=len(dim1_indices), desc='Processing dim1') as pbar:
        fig_width = max(2.0, data.shape[2] / 20.0)
        fig_height = max(2.0, data.shape[0] / 20.0)
        for i in dim1_indices:
            save_slice_image(data[:, i, :], f'{filename}/X-{i}.png', fig_width, fig_height, imshow_params)
            pbar.update(1)

    # dim2 -> Y-i
    dim2_indices = list(range(0, data.shape[2], step))
    with tqdm(total=len(dim2_indices), desc='Processing dim2') as pbar:
        fig_width = max(2.0, data.shape[1] / 20.0)
        fig_height = max(2.0, data.shape[0] / 20.0)
        for i in dim2_indices:
            save_slice_image(data[:, :, i], f'{filename}/Y-{i}.png', fig_width, fig_height, imshow_params)
            pbar.update(1)
