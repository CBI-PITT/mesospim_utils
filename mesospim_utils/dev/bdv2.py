# pip install h5py

import h5py
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from itertools import product
from skimage import io


class Mesospim_BDV_Reader:
    _f = None
    def __init__(self, path: Path, ResolutionLevelLock=0):
        assert isinstance(path, (Path, str)), 'BDV_Reader must be given a string or Path object'

        if not isinstance(path, Path):
            path = Path(path)

        assert path.suffix == '.xml', 'BDV_Reader path must be to a .xml file'

        self.bdv_xml = path

        self.parse_bdv_metadata()  # Adds self._relative_image_file, self.transforms, self.voxel_info
        # self._relative_image_file    ## Name of .h5 file relative to .xml as defined self.bdv_xml
        # self.transforms              ## dict of numpy affine transforms for each tile, keys=int_tile_num
        # self.voxel_info              ## dict[dict], keys ['size','unit'], 'size' = resolution in 'unit'
        # self.Channels                ## number of channels

        self.bdv_file = self.bdv_xml.parent / self._relative_image_file

        self.dir = self.bdv_file.parent

        self.ResolutionLevelLock = ResolutionLevelLock
        self._resolution0 = tuple(self.voxel_info[0]['size'])
        self.ResolutionUnits = self.voxel_info[0]['unit']

        self.TileLock = 0

        with h5py.File(self.bdv_file, "r") as f:
            TileList = list(f.keys())
            self.TileList = [x for x in TileList if 't00' not in x]
            self.TileNum = len(self.TileList)
            sample_tile = self.TileList[0]

            self.ResolutionLevels = len(f[sample_tile]['resolutions'])
            self.ResolutionSampling = [tuple(f[sample_tile]['resolutions'][x].tolist()[::-1]) for x in
                                       range(self.ResolutionLevels)]  # [::-1] -> (z,y,x)
            self.ChunkShapes = [tuple(f[sample_tile]['subdivisions'][x].tolist()[::-1]) for x in
                                range(self.ResolutionLevels)]  # [::-1] -> (z,y,x)

            dset_path = self._get_dset_path(0, 0)
            self.TileShapes = [f[self._get_dset_path(0, res)].shape for res in range(self.ResolutionLevels)]
            self.dtype = f[dset_path].dtype
            self.ndim = f[dset_path].ndim

        self._get_tile_per_channel()
        # self._tiles_by_channel        ## dict[ch] = {1,2,3,...}

        self.get_transforms_relative_to_anchor()
        self._collect_multiscale_transforms()
        self.set_resolution_lock(self.ResolutionLevelLock)
        # self.multiscale_transforms[res][tile]
        # self.compute_bounding_box()

    def _get_tile_per_channel(self):
        tiles = list(range(self.TileNum))
        self._tiles_by_channel = {x : [] for x in range(self.Channels)}
        for ch in range(self.Channels):
            tiles_per_channel = self.TileNum // self.Channels
            tiles_start = tiles_per_channel * ch
            tiles_stop = tiles_start + tiles_per_channel
            self._tiles_by_channel[ch] = tiles[tiles_start:tiles_stop]
        return self._tiles_by_channel




    def set_resolution_lock(self, res_level: int):
        if isinstance(res_level, str):
            res_level = int(res_level)

        assert res_level >= 0 < self.ResolutionLevels, f'Resolution level {res_level} does not exist'

        self.shape = self.TileShapes[res_level]
        self.chunks = self.ChunkShapes[res_level]
        self.resolution = tuple([x * y for x, y in zip(self._resolution0, self.ResolutionSampling[res_level])])
        self.ResolutionLevelLock = res_level
        self.compute_bounding_box()


    def set_tile_lock(self, tile_num: int):
        if isinstance(tile_num, str):
            tile_num = int(tile_num)

        assert tile_num >= 0 < self.TileNum, f'Tile number {tile_num} does not exist'

        self.TileLock = tile_num


    def _get_dset_path(self, tile_num: int, res_level: int):
        tile = self.TileList[tile_num]
        return f't00000/{tile}/{res_level}/cells'


    def _open_bdf_file(self):
        with h5py.File(self.bdv_file, "r") as f:
            return f


    def __enter__(self):
        self._f = h5py.File(self.bdv_file, "r")
        return self._f


    def __exit__(self, exc_type, exc_value, traceback):
        if self._f is not None:
            self._f.close()
            self._f = None
        # Return False to propagate exceptions, True to suppress
        return False


    # def __getitem__(self, item):
    #     dset_path = self._get_dset_path(self.TileLock, self.ResolutionLevelLock)
    #     print(dset_path)
    #     with self as f:
    #         return f[dset_path][item]


    def parse_bdv_metadata(self):
        tree = ET.parse(self.bdv_xml)
        root = tree.getroot()

        voxel_info = {}
        transforms = {}
        image_file = None

        # Extract image loader filename
        hdf5_elem = root.find('.//ImageLoader/hdf5')
        if hdf5_elem is not None:
            image_file = hdf5_elem.text.strip()

        # Extract voxel sizes and units
        for vs in root.findall('.//ViewSetup'):
            setup_id = int(vs.find('id').text)
            unit = vs.find('./voxelSize/unit').text
            size_text = vs.find('./voxelSize/size').text.strip()
            size = list(map(float, size_text.split()))
            voxel_info[setup_id] = {'unit': unit, 'size': size[::-1]}  # [::-1] -> (z,y,x)

        # Extract Translation to Regular Grid and convert to affine matrix (for z, y, x indexing)
        for vr in root.findall('.//ViewRegistration'):
            setup_id = int(vr.attrib['setup'])
            for vt in vr.findall('./ViewTransform'):
                name = vt.find('Name').text
                if name == "Translation to Regular Grid":
                    affine_text = vt.find('affine').text.replace('\n', ' ').strip()
                    values = list(map(float, affine_text.split()))
                    matrix_3x4 = np.array(values).reshape(3, 4)

                    # # !TODO need to determine how to deal with the other tranforms (ie rot, sheer)
                    # # Reorder axes (for translation) from (x, y, z) to (z, y, x)
                    # new_translation = matrix_3x4[:,-1][2::-1]
                    # reordered = matrix_3x4.copy()
                    # reordered[0,-1] = new_translation[0]
                    # reordered[1, -1] = new_translation[1]
                    # reordered[2, -1] = new_translation[2]

                    # Form full 4x4 affine
                    matrix_4x4 = np.eye(4)
                    # matrix_4x4[:3, :] = reordered
                    matrix_4x4[:3, :] = matrix_3x4
                    matrix_4x4 = self.invert_translation_xy(matrix_4x4) # Negative translation to positive
                    transforms[setup_id] = self.reorder_affine_for_zyx(matrix_4x4) # xyz -> zyx

        channel_map = {}
        for channel in root.findall(".//Attributes[@name='channel']/Channel"):
            cid = int(channel.find("id").text)
            cname = channel.find("name").text.strip()
            channel_map[cid] = cname

        self.Channels = len(channel_map)
        self.voxel_info = voxel_info
        self.transforms = transforms
        self._relative_image_file = Path(image_file)


    def compute_bounding_box(self, anchor_id=0):
        all_corners = []

        shape = self.shape

        # Define 8 corner coordinates of a volume in z,y,x order
        zyx_corners = np.array([
            [0, 0, 0],
            [0, 0, shape[2]],
            [0, shape[1], 0],
            [0, shape[1], shape[2]],
            [shape[0], 0, 0],
            [shape[0], 0, shape[2]],
            [shape[0], shape[1], 0],
            [shape[0], shape[1], shape[2]],
        ])

        # Convert to homogeneous coordinates
        zyx_corners_h = np.hstack([zyx_corners, np.ones((8, 1))])  # shape (8, 4)

        # Invert the anchor's affine matrix
        anchor_inv = np.linalg.inv(self.multiscale_transforms[self.ResolutionLevelLock][anchor_id])

        all_transformed_points = []

        for setup_id, affine in self.multiscale_transforms[self.ResolutionLevelLock].items():
            # Map voxel corners into global space for this tile
            global_corners = affine @ zyx_corners_h.T  # shape (4, 8)
            # Map those points into anchor-relative space
            anchor_relative = (anchor_inv @ global_corners).T[:, :3]
            all_transformed_points.append(anchor_relative)

        all_transformed_points = np.vstack(all_transformed_points)
        bbox_min = np.min(all_transformed_points, axis=0)
        bbox_max = np.max(all_transformed_points, axis=0)

        self._min_coords = tuple(bbox_min.tolist())
        self._max_coords = tuple(bbox_max.tolist())
        self._canvas_shape = [int(x//1) for x in self._max_coords]


    def get_transforms_relative_to_anchor(self, anchor_id=0):
        """
        Return a new dictionary of transforms and inverse transforms where anchor_id becomes the identity matrix,
        and all other transforms are translated relative to it.
        """
        anchor_offsets = 0 - self.transforms[anchor_id][:3, -1]
        new_transforms = {}
        new_inverse_transforms = {}

        for setup_id, affine in self.transforms.items():
            new_translation = affine[:3, -1] + anchor_offsets
            new_affine = affine.copy()
            new_affine[:3, -1] = new_translation

            ## Floor function truncated affines to integer translation for pixels
            new_affine = np.floor(new_affine)
            new_transforms[setup_id] = new_affine
            new_inverse_transforms[setup_id] = np.linalg.inv(new_affine)

        self.relative_transforms = new_transforms
        self.relative_transforms_inv = new_inverse_transforms


    def reorder_affine_for_zyx(self,original_affine):
        perm = [2, 1, 0]  # x->z, y->y, z->x

        R = original_affine[:3, :3]
        t = original_affine[:3, 3]

        R_new = R[perm, :][:, perm]
        t_new = t[perm]

        reordered = np.eye(4)
        reordered[:3, :3] = R_new
        reordered[:3, 3] = t_new

        return reordered

    def invert_translation_xy(self,origional_affine):
        origional_affine[: 3, -1] *= -1
        return origional_affine

    def resample_translation(self, affine, sample_factor: tuple=(1,2,2)):
        origional_translation = affine[: 3, -1]
        new_translation = origional_translation // np.array(sample_factor)
        new_transform = affine.copy()
        new_transform[: 3, -1] = new_translation
        return new_transform

    def _collect_multiscale_transforms(self):
        self.multiscale_transforms = {x:{} for x in range(self.ResolutionLevels)}
        self.multiscale_transforms_inv = {x: {} for x in range(self.ResolutionLevels)}
        for res in range(self.ResolutionLevels):
            print(f'{self.relative_transforms=}')
            for tile, trans in self.relative_transforms.items():
                self.multiscale_transforms[res][tile] = self.resample_translation(trans,self.ResolutionSampling[res])
                self.multiscale_transforms_inv[res][tile] = np.linalg.inv(self.multiscale_transforms[res][tile])



    ##########################################################################################################
    ## BUILOING MONTAGE ARRAY ##
    ##########################################################################################################

    def _slice_to_bbox(self,slc, shape=None):
        """
        Convert a slice or tuple of slices to a bounding box in (min_z, min_y, min_x, max_z, max_y, max_x) format.

        Parameters:
            slc (slice or tuple of slices): Slicing of the 3D volume.
            shape (tuple, optional): Shape of the array for interpreting `None` or negative indices.

        Returns:
            tuple: (min_z, min_y, min_x, max_z, max_y, max_x)
        """
        if isinstance(slc, slice):
            slc = (slc,)

        # Extend to full 3D if needed
        slc = tuple(slc) + (slice(None),) * (3 - len(slc))

        starts = []
        stops = []
        for i, s in enumerate(slc):
            start, stop, step = s.indices(shape[i] if shape else 2 ** 31 - 1)
            starts.append(start)
            stops.append(stop)

        return tuple(starts) + tuple(stops)


    def _transform_bbox(self, bbox, affine):
        """
        Transforms a 3D bounding box using an affine transform matrix.

        Parameters:
            bbox (tuple): (min_z, min_y, min_x, max_z, max_y, max_x)
            matrix (ndarray): 4x4 affine transform matrix (homogeneous coords)

        Returns:
            tuple: Transformed bounding box (min_z, min_y, min_x, max_z, max_y, max_x)
        """
        min_z, min_y, min_x, max_z, max_y, max_x = bbox

        # Generate 8 corners of the bbox
        corners = np.array(list(product(
            [min_z, max_z],
            [min_y, max_y],
            [min_x, max_x]
        )))  # shape (8, 3)

        # Convert to homogeneous coordinates (Z, Y, X, 1)
        ones = np.ones((corners.shape[0], 1))
        corners_hom = np.hstack([corners, ones])  # shape (8, 4)

        # Apply affine transform
        transformed = (affine @ corners_hom.T).T[:, :3]  # shape (8, 3)

        # Get min/max along each axis
        min_vals = transformed.min(axis=0)
        max_vals = transformed.max(axis=0)

        # # Floor values
        # min_vals = np.floor(min_vals)
        # max_vals = np.ceil(max_vals)
        # print(min_vals)
        # print(max_vals)

        # Return as bounding box
        return tuple(min_vals.tolist() + max_vals.tolist())

    def _bboxes_overlap(self, bbox1, bbox2):
        """
        Check if two 3D bounding boxes overlap.

        Parameters:
            bbox1 (tuple): (min_z, min_y, min_x, max_z, max_y, max_x)
            bbox2 (tuple): (min_z, min_y, min_x, max_z, max_y, max_x)

        Returns:
            bool: True if the bounding boxes overlap, False otherwise.
        """
        for i in range(3):
            min1, max1 = bbox1[i], bbox1[i + 3]
            min2, max2 = bbox2[i], bbox2[i + 3]

            # Check for no overlap along this axis
            if max1 <= min2 or max2 <= min1:
                return False

        return True

    def _bbox_to_slice(self,bbox, shape=None, round_mode='floor'):
        """
        Convert a bounding box into a tuple of slices, optionally clipping to shape.

        Parameters:
            bbox (tuple): (min_z, min_y, min_x, max_z, max_y, max_x), in subarray coordinates.
            shape (tuple, optional): If provided, clips bbox to this shape (z, y, x).
            round_mode (str): 'floor' to floor mins and ceil maxs, or 'round' to nearest.

        Returns:
            tuple of slice: (slice_z, slice_y, slice_x)
        """
        import math

        slices = []
        for i in range(3):
            min_v = bbox[i]
            max_v = bbox[i+3]

            if round_mode == 'floor':
                start = int(np.floor(min_v))
                stop = int(np.ceil(max_v))
                # stop = int(np.floor(max_v))
            elif round_mode == 'round':
                start = int(round(min_v))
                stop = int(round(max_v))
            else:
                raise ValueError(f"Unsupported round_mode: {round_mode}")

            if shape:
                start = max(0, min(start, shape[i]))
                stop = max(0, min(stop, shape[i]))

            slices.append(slice(start, stop))

        return tuple(slices)

    def _intersecting_bbox(self, bbox1, bbox2):

        mins = []
        maxs = []
        for i in range(3):
            low = max(bbox1[i],bbox2[i])
            high = min(bbox1[i+3],bbox2[i+3])
            mins.append(low)
            maxs.append(high)

        return tuple(mins) + tuple(maxs)

    #########  WORKING GETITEM ##################
    # def __getitem__(self, item):
    #     # t = tile
    #     # c = canvas
    #     # -------------
    #     # v = virtual space
    #     # n = native (original) space
    #     # -------------
    #     # w = whole
    #     # r = roi
    #     # bbox = bounding box (minz, miny, minx, maxz, maxy, maxx)
    #
    #     blend = 'max'
    #
    #     t_n_w_bbox = (0,0,0,*self.shape)
    #     c_v_r_bbox = self._slice_to_bbox(item, shape=self._canvas_shape) #In Virt array
    #     array = np.zeros((c_v_r_bbox[3] - c_v_r_bbox[0], c_v_r_bbox[4] - c_v_r_bbox[1], c_v_r_bbox[5] - c_v_r_bbox[2]),
    #                      dtype=self.dtype)
    #
    #     if blend == 'mean':
    #         log_array = np.zeros((c_v_r_bbox[3] - c_v_r_bbox[0], c_v_r_bbox[4] - c_v_r_bbox[1], c_v_r_bbox[5] - c_v_r_bbox[2]),
    #                          dtype='uint8')
    #     # print(bbox)
    #     # for tile in range(self.TileNum):
    #     for tile in range(self.TileNum//2): # HACK TO GET ONLY FIRST CHANNEL
    #         affine = self.relative_transforms[tile]
    #         affine_inv = self.relative_transforms_inv[tile]
    #
    #         t_v_w_bbox = self._transform_bbox(t_n_w_bbox, affine)
    #         # print(subarray_bbox)
    #         overlap = self._bboxes_overlap(t_v_w_bbox,c_v_r_bbox)
    #         print(f'Overlap is: {overlap} for Tile{tile}')
    #         if not overlap:
    #             continue
    #
    #         t_v_r_bbox = self._intersecting_bbox(t_v_w_bbox,c_v_r_bbox)
    #         print(f'{t_v_r_bbox=}')
    #         t_n_r_bbox = self._transform_bbox(t_v_r_bbox, affine_inv)
    #         t_n_r_slice = self._bbox_to_slice(t_n_r_bbox)
    #         print(f'{t_n_r_slice=}')
    #
    #         t_v_r_bbox = [np.floor(x) for x in t_v_r_bbox]
    #         c_v_r_bbox = [np.floor(x) for x in c_v_r_bbox]
    #         c_n_r_bbox = (t_v_r_bbox[0] - c_v_r_bbox[0],
    #                       t_v_r_bbox[1] - c_v_r_bbox[1],
    #                       t_v_r_bbox[2] - c_v_r_bbox[2],
    #                       t_v_r_bbox[3] - c_v_r_bbox[0],
    #                       t_v_r_bbox[4] - c_v_r_bbox[1],
    #                       t_v_r_bbox[5] - c_v_r_bbox[2]
    #                       )
    #
    #         # c_n_r_bbox = tuple(
    #         #     float(np.round(t_v_r_bbox[i] - c_v_r_bbox[i % 3], decimals=5))
    #         #     for i in range(6)
    #         # )
    #
    #         # virt_array_slice = self._transform_bbox(sub_array_roi, affine)
    #         c_n_r_slice = self._bbox_to_slice(c_n_r_bbox)
    #         print(f'{c_n_r_slice=}')
    #
    #         print(f'Array Shape = {array.shape}')
    #         with self as f:
    #             dset_path = self._get_dset_path(tile, self.ResolutionLevelLock)
    #             print(dset_path)
    #             if blend == 'max':
    #                 array[c_n_r_slice] = np.maximum(
    #                     f[dset_path][t_n_r_slice],
    #                     array[c_n_r_slice]
    #                 )
    #             elif blend == 'mean':
    #                 log_array[c_n_r_slice] += 1
    #                 array[c_n_r_slice] += f[dset_path][t_n_r_slice]
    #
    #     if blend == 'mean':
    #         array = np.true_divide(array,log_array)
    #     return array

    def _slice_fixer(self, index, ndim=5):
        """
        Normalize an indexing object (int, slice, ellipsis, etc.) into a tuple
        of fully-formed slices, suitable for reverse-dimension logic.
        """
        if not isinstance(index, tuple):
            index = (index,)

        normalized = []
        has_ellipsis = False

        for item in index:
            if item is Ellipsis:
                if has_ellipsis:
                    raise IndexError("an index can only have a single ellipsis ('...')")
                has_ellipsis = True
                # Calculate how many slices to expand
                num_to_add = ndim - (len(index) - 1)
                normalized.extend([slice(None)] * num_to_add)
            elif item is None:
                normalized.append(slice(None))  # np.newaxis
            elif isinstance(item, int):
                normalized.append(slice(item, item + 1, 1))
            elif isinstance(item, slice):
                normalized.append(item)
            else:
                raise TypeError(f"Unsupported index type: {type(item)}")

        # If no ellipsis was found and it's still shorter than ndim, pad from the left
        if len(normalized) < ndim:
            normalized = normalized + [slice(None)] * (ndim - len(normalized))

        return tuple(normalized)

    def _slice_length(self, slc, dim_size):
        """
        Return the number of elements a slice selects, given the total size of the dimension.

        Args:
            slc (slice): The slice object.
            dim_size (int): The size of the array dimension being sliced.

        Returns:
            int: The number of selected elements.
        """
        start, stop, step = slc.indices(dim_size)
        return max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)

    def __getitem__(self, item):
        # t = tile
        # c = canvas
        # -------------
        # v = virtual space
        # n = native (original) space
        # -------------
        # w = whole
        # r = roi
        # bbox = bounding box (minz, miny, minx, maxz, maxy, maxx)

        item = self._slice_fixer(item) # Assumes full 5-axis slicing (t,c,z,y,x)
        print(item)
        time_points = range(1)[item[0]] # Lock time point 1 for mesospim may expand later
        channels = range(self.Channels)[item[1]]
        item = item[2:]


        num_time = len(time_points)
        num_ch = len(channels)


        blend = 'max'

        t_n_w_bbox = (0,0,0,*self.shape)
        c_v_r_bbox = self._slice_to_bbox(item, shape=self._canvas_shape) #In Virt array
        array = np.zeros((num_time,num_ch,c_v_r_bbox[3] - c_v_r_bbox[0], c_v_r_bbox[4] - c_v_r_bbox[1], c_v_r_bbox[5] - c_v_r_bbox[2]),
                         dtype=self.dtype)

        if blend == 'mean':
            log_array = np.zeros((num_time,num_ch,c_v_r_bbox[3] - c_v_r_bbox[0], c_v_r_bbox[4] - c_v_r_bbox[1], c_v_r_bbox[5] - c_v_r_bbox[2]),
                             dtype='uint8')

        for t in time_points:
            for c in channels:

                current_tiles = self._tiles_by_channel[c]
                for tile in current_tiles:
                    affine = self.multiscale_transforms[self.ResolutionLevelLock][tile]
                    affine_inv = self.multiscale_transforms_inv[self.ResolutionLevelLock][tile]

                    t_v_w_bbox = self._transform_bbox(t_n_w_bbox, affine)
                    # print(subarray_bbox)
                    overlap = self._bboxes_overlap(t_v_w_bbox,c_v_r_bbox)
                    print(f'Overlap is: {overlap} for Tile{tile}')
                    if not overlap:
                        continue

                    t_v_r_bbox = self._intersecting_bbox(t_v_w_bbox,c_v_r_bbox)
                    print(f'{t_v_r_bbox=}')
                    t_n_r_bbox = self._transform_bbox(t_v_r_bbox, affine_inv)
                    t_n_r_slice = self._bbox_to_slice(t_n_r_bbox)
                    print(f'{t_n_r_slice=}')

                    t_v_r_bbox = [np.floor(x) for x in t_v_r_bbox]
                    c_v_r_bbox = [np.floor(x) for x in c_v_r_bbox]
                    c_n_r_bbox = (t_v_r_bbox[0] - c_v_r_bbox[0],
                                  t_v_r_bbox[1] - c_v_r_bbox[1],
                                  t_v_r_bbox[2] - c_v_r_bbox[2],
                                  t_v_r_bbox[3] - c_v_r_bbox[0],
                                  t_v_r_bbox[4] - c_v_r_bbox[1],
                                  t_v_r_bbox[5] - c_v_r_bbox[2]
                                  )

                    c_n_r_slice = self._bbox_to_slice(c_n_r_bbox)
                    print(f'{c_n_r_slice=}')

                    print(f'Array Shape = {array.shape}')
                    with self as f:
                        dset_path = self._get_dset_path(tile, self.ResolutionLevelLock)
                        print(dset_path)
                        if blend == 'max':
                            array[(t, c) + c_n_r_slice] = np.maximum(
                                f[dset_path][t_n_r_slice],
                                array[(t, c) + c_n_r_slice]
                            )
                        elif blend == 'mean':
                            log_array[(t, c) + c_n_r_slice] += 1
                            array[(t, c) + c_n_r_slice] += f[dset_path][t_n_r_slice]

                if blend == 'mean':
                    array = np.true_divide(array,log_array)
        print(array.shape)
        print(f'{t=}')
        print(f'{c=}')
        array = array.squeeze()
        print(array.shape)
        return array


if __name__ == '__main__':
    bdv_file = '/CBI_FastStore/Acquire/MesoSPIM/alan_test/test_h5_Mag4x_Tile0_ch488_bdv.xml'
    bdv_file = Path(bdv_file)

    z = Mesospim_BDV_Reader(bdv_file)

    # Create an example image (a 2D NumPy array)
    # image = z[100:101, 1024:3096:, 1024:3096]
    shape_start = z._canvas_shape[1]//2
    shape_stop = shape_start + 1024

    # image = z[0, :, 750:755, :, :]
    z.set_resolution_lock(2)
    image = z[0,0]
    image = image.astype('uint16')
    # image *= 100

    # Save the image as a TIFF file
    io.imsave('/CBI_FastStore/tmp/2025-05.tif', image)
