### Toolkit-nuScenes

We provide the toolkit of generating noisy validation pkl based MMDetection3D.

Please following MMDetection3D to download and organize the [nuScenes Dataset](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/zh_cn/datasets/nuscenes_det.md) .

### Noise validation pkl generation:

Note：You can generate noisy validation pkl files following the instructions or download it from [[GitHub Release](https://github.com/anonymous-benchmark/lidar-camera-robust-benchmark/releases/tag/publish)].

```python
python tools/create_noise_data_nuscenes.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

##### the format of pkl files：

```python
dict(
	'lidar': dict(...) # noisy infos of lidar, retrieved by LiDAR file name
  'camera': dict(...) # noisy infos of camera, retrieved by camera file name
)
```

For 'lidar'，the format is:

```python
dict(
	'xxxx.bin': dict(...),
    'xxxx.bin': dict(...)
)
# the content of above dict(...):
dict(
    # basic information
	'prev': 'xxxx.bin' or '' # LiDAR file name of the previous frame. If no previous frame, then the file name is ''.
    'cam': dict('CAM_FRONT':'xxx.jpg', 'CAM_FRONT_RIGHT':'xxx.jpg', ...) # corresponding camera file name.
    'mmdet': dict() # basic information from MMDetection3D, including sweeps.
    # Noisy information
    'noise': dict(
    	'drop_frames': dict(   # The information of LiDAR-stuck. The key is the percentage, including 10, 20, ..., 90.
        	'10': dict('discrete': dict('stuck': True or False # if stuck? 
                                      'replace': 'xxx.bin' #the replaced LiDAR file
                       'consecutive': dict('stuck': True or False, 'replace': 'xxx.bin' # same as the above
                  )
            '20': # same as the above
            ...
            '90': # same as the above
        ),
        'object_failure': True/False    
    )
)
```



For 'camera'，the format is similar to 'lidar' part:

```python
dict(
	'xxxx.jpg': dict(...),
    'xxxx.jpg': dict(...)
)
# the content of above dict(...):
dict(
    # basic information
    'type': 'CAM_FRONT' or 'CAM_FRONT_RIGHT' or ... # the camera type
	'prev': 'xxxx.jpg' or '' # Camera file name of the previous frame. If no previous frame, then the file name is ''.
    'lidar': dict('file_name': 'xxx.bin') # corresponding LiDAR name
    # Noisy information
    'noise': dict(
    	'drop_frames': dict(   # The information of camera-stuck. The key is the percentage, including 10, 20, ..., 90.
        	'10': dict('discrete': dict('stuck': True or False # if stuck? 
                                      'replace': 'xxx.jpg' # the replaced camera file
                       'consecutive': dict('stuck': True or False, 'replace': 'xxx.jpg' # same as the above
                  )
            '20': # same as the above
            ...
            '90': # same as the above
        )
        'extrinsics_noise': dict(
            'sensor2ego_translation': xxx, # original translation matrix
        	'single_noise_sensor2ego_translation': xxx, # the noisy translation matrix. 'single' means the noisy is independent for all cameras 
            'all_noise_sensor2ego_translation': xxx, # the noisy translation matrix. 'all' means the noisy is the same for all cameras 
            ...
        ),
        'mask_noise': dict(
            'mask_id': xxx, # the mask image ID
        )
    )
)
```



### nuScenesNoiseDataset

Usage：replace the init and get_data_info function of  NuScenesDataset in  datasets/nuscenes_dataset.py with the following code.

```python
@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    def __init__(self,
                 ann_file,
                 num_views=6,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 # Add
                 noise_nuscenes_ann_file = '',
                 extrinsics_noise=False,
                 extrinsics_noise_type='single',
                 drop_frames=False,
                 drop_set=[0,'discrete'],
                 noise_sensor_type='camera'):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.num_views = num_views
        assert self.num_views <= 6
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        ## ADD
        self.extrinsics_noise = extrinsics_noise # if use noisy calib
        assert extrinsics_noise_type in ['all', 'single'] 
        self.extrinsics_noise_type = extrinsics_noise_type # single or all
        self.drop_frames = drop_frames # if use lidar-stuck or camera-stuck
        self.drop_ratio = drop_set[0] # the percentage：assert ratio in [10, 20, ..., 90]
        self.drop_type = drop_set[1] # consecutive or discrete
        self.noise_sensor_type = noise_sensor_type # lidar or camera

        if self.extrinsics_noise or self.drop_frames:
            noise_data = mmcv.load(noise_nuscenes_ann_file, file_format='pkl')
            self.noise_data = noise_data[noise_sensor_type]
        else:
            self.noise_data = None
        
        print('noise setting:')
        if self.drop_frames:
            print('frame drop setting: drop ratio:', self.drop_ratio, ', sensor type:', self.noise_sensor_type, ', drop type:', self.drop_type)
        if self.extrinsics_noise:
            assert noise_sensor_type=='camera'
            print(f'add {extrinsics_noise_type} noise to extrinsics')
        
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
		# ADD
        if self.noise_sensor_type == 'lidar':
            if self.drop_frames:
                pts_filename = input_dict['pts_filename']
                file_name = pts_filename.split('/')[-1]

                if self.noise_data[file_name]['noise']['drop_frames'][self.drop_ratio][self.drop_type]['stuck']:
                    replace_file = self.noise_data[file_name]['noise']['drop_frames'][self.drop_ratio][self.drop_type]['replace']
                    if replace_file != '':
                        pts_filename = pts_filename.replace(file_name, replace_file)

                        input_dict['pts_filename'] = pts_filename
                        input_dict['sweeps'] = self.noise_data[replace_file]['mmdet_info']['sweeps']
                        input_dict['timestamp'] = self.noise_data[replace_file]['mmdet_info']['timestamp'] / 1e6

        cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            caminfos = []
            # for cam_type, cam_info in info['cams'].items():
            for cam_type in cam_orders:
                cam_info = info['cams'][cam_type]

                cam_data_path = cam_info['data_path']
                file_name = cam_data_path.split('/')[-1]
                if self.noise_sensor_type == 'camera':
                    if self.drop_frames:
                        if self.noise_data[file_name]['noise']['drop_frames'][self.drop_ratio][self.drop_type]['stuck']:
                            replace_file = self.noise_data[file_name]['noise']['drop_frames'][self.drop_ratio][self.drop_type]['replace']
                            if replace_file != '':
                                cam_data_path = cam_data_path.replace(file_name, replace_file)

                                # print(file_name, self.noise_data[file_name]['prev'])

                image_paths.append(cam_data_path)
                # obtain lidar to image transformation matrix
                if self.extrinsics_noise:
                    sensor2lidar_rotation = self.noise_data[file_name]['noise']['extrinsics_noise'][f'{self.extrinsics_noise_type}_noise_sensor2lidar_rotation']
                    sensor2lidar_translation = self.noise_data[file_name]['noise']['extrinsics_noise'][f'{self.extrinsics_noise_type}_noise_sensor2lidar_translation']
                else:
                    sensor2lidar_rotation = cam_info['sensor2lidar_rotation']
                    sensor2lidar_translation = cam_info['sensor2lidar_translation']

                lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
                lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                caminfos.append(
                    {'sensor2lidar_translation':sensor2lidar_translation, 
                    'sensor2lidar_rotation':sensor2lidar_rotation})

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    caminfo=caminfos
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
```



### Limited LiDAR FOV

Usage：replace LoadPointsFromMultiSweeps in datasets/pipelines/loading.py with the following code. Then, you should add parameters point_cloud_angle_range=[-90, 90] or [-60,60] in config files.

```python
@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False,
                 point_cloud_angle_range=None):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

        if point_cloud_angle_range is not None:
            self.filter_by_angle = True
            self.point_cloud_angle_range = point_cloud_angle_range
            print(point_cloud_angle_range)
        else:
            self.filter_by_angle = False
            # self.point_cloud_angle_range = point_cloud_angle_range

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]
    
    def filter_point_by_angle(self, points):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        # print(points_numpy[points_numpy[:,1]>0])
        pts_phi = (np.arctan(points_numpy[:, 0] / points_numpy[:, 1]) + (points_numpy[:, 1] < 0) * np.pi + np.pi * 2) % (np.pi * 2) 
        
        pts_phi[pts_phi>np.pi] -= np.pi * 2
        pts_phi = pts_phi/np.pi*180
        
        assert np.all(-180 <= pts_phi) and np.all(pts_phi <= 180)

        filt = np.logical_and(pts_phi>=self.point_cloud_angle_range[0], pts_phi<=self.point_cloud_angle_range[1])

        return points[filt]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray): Multi-sweep point cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        if self.filter_by_angle:
            points = self.filter_point_by_angle(points)

        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'
```



### Missing of Camera Inputs

Usage：replace LoadMultiViewImageFromFiles in datasets/pipelines/loading.py with the following code. Then, you should add parameters drop_camera=[] in config files, like drop_camera=['CAM_FRONT']. The names in [] indicate the types of missing cameras. 

```python
@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged',
                 # ADD
                 drop_camera=[]):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        # ADD
        self.cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        self.drop_camera = drop_camera

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            img = np.concatenate([img, np.zeros_like(img[0:1280-886,:])], axis=0)
        return img

    def __call__(self, results):
        filename = results['img_filename']

        # ADD
        img_lists = []
        for i, name in enumerate(filename):
            single_img = mmcv.imread(name, self.color_type)
            if self.img_scale is not None:
                single_img = self.pad(single_img)
            if self.cam_orders[i] in self.drop_camera:
                img_lists.append(np.zeros_like(single_img))
            else:
                img_lists.append(single_img)
        img = np.stack(img_lists, axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)



        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = img_lists
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = [1.0, 1.0]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)

```



### Occlusion of Camera Lens 

please download the mask images from [[GitHub Release](https://github.com/anonymous-benchmark/lidar-camera-robust-benchmark/releases/tag/publish)] first.

```python
@PIPELINES.register_module()
class LoadMaskMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged',
                 # ADD
                 noise_nuscenes_ann_file='', mask_file=''):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        
        # ADD
        noise_data = mmcv.load(noise_nuscenes_ann_file, file_format='pkl')
        self.noise_camera_data = noise_data['camera']
        self.mask_file = mask_file

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            img = np.concatenate([img, np.zeros_like(img[0:1280-886,:])], axis=0)
        return img
    
    # ADD
    def put_mask_on_img(self, img, mask):
        h, w = img.shape[:2]
        mask = np.rot90(mask)
        mask = mmcv.imresize(mask, (w, h), return_scale=False)
        alpha = mask / 255
        alpha = np.power(alpha, 3)
        img_with_mask = alpha * img + (1 - alpha) * mask

        return img_with_mask

    def __call__(self, results):
        filename = results['img_filename']

        img_lists = []
        for name in filename:
            single_img = mmcv.imread(name, self.color_type)
            if self.img_scale is not None:
                single_img = self.pad(single_img)
            # ADD
            noise_index = name.split('/')[-1]
            mask_id_png = 'mask_'+ str(self.noise_camera_data[noise_index]['noise']['mask_noise']['mask_id']) + '.jpg'
            mask_name = os.path.join(self.mask_file, mask_id_png)
            mask = mmcv.imread(mask_name, self.color_type)
            single_img = self.put_mask_on_img(single_img, mask)
            img_lists.append(single_img)
        img = np.stack(img_lists, axis=-1)
        
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = [1.0, 1.0]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)
```



### LiDAR Object Failure

Usage：add a new class Randomdropforeground in datasets/pipelines/transforms_3d.py (do not forget to add import in datasets/pipelines/init.py). Then, in the config files, you should add Randomdropforeground in the transforms of test_pipeline. Besides, you also need to add dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True).

```python
@PIPELINES.register_module()
class Randomdropforeground(object):
    def __init__(self, noise_nuscenes_ann_file=''):
        noise_data = mmcv.load(noise_nuscenes_ann_file, file_format='pkl')
        self.noise_lidar_data = noise_data['lidar']

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.
        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.
        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        pts_filename = input_dict['pts_filename']
        noise_index = pts_filename.split('/')[-1]

        points = input_dict['points']
        if self.noise_lidar_data[noise_index]['noise']['object_failure']:
            points = self.remove_points_in_boxes(points, gt_bboxes_3d.tensor.numpy())
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += ' fore_drop_rate={})'.format(self.drop_rate)
        return repr_str
```