import os
import sys
import pdb
import time
import math
import json
import psutil
import logging
import pydicom
import matplotlib
import numpy as np
from pubsub import pub
import multiprocessing
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)

class Processor:
    def __init__(self):
        self.reset()
        pub.subscribe(self.reset, 'reset')
        pub.subscribe(self.listener, 'incoming')
        pub.subscribe(self.snr_controller, 'proc-snr')
        multiprocessing.set_start_method('fork')


    def reset(self):
        self._instances = SortedDict()
        self._slice_means = SortedDict()
        pub.sendMessage('plot_snr', snr_metric=str(0.0))
        logger.debug('received message to reset')

    def listener(self, ds, path):
        self._key = int(ds.InstanceNumber)
        self._instances[self._key] = {
            'path': path,
            'volreg': None,
            'nii_path': None
        }
        self._slice_means[self._key] = {
            'path': path,
            'slice_means': None,
            'mask_threshold': None,
            'mask': None
        }
        logger.debug('current state of instances')
        logger.debug(json.dumps(self._instances, default=list, indent=2))

        num_vols = ds[(0x0020, 0x0105)].value

        data_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))

        num_dicoms = len([name for name in os.listdir(data_path) if name.endswith('.dcm')])

        if num_dicoms > 0:
            self._snr_interval = 1
        else:
            self._snr_interval = 1


        logger.info(f'CURRENT SNR INTERVAL {self._snr_interval}')

        #if self._key == :
        #    self._plot_dict = dict.fromkeys(range(1, num_vols+1), None)


        #self._plot_dict[self._key] = num_dicoms

        #self._plot_dict[self._key] = psutil.cpu_percent()

        tasks = self.check_volreg(self._key)

        snr_tasks = self.check_snr(self._key)
        logger.debug('publishing message to volreg topic with the following tasks')
        logger.debug(json.dumps(tasks, indent=2))
        pub.sendMessage('volreg', tasks=tasks)
        logger.debug(f'publishing message to params topic')
        pub.sendMessage('params', ds=ds)
        logger.debug(f'publishing message to snr topic')
        logger.debug(f'snr task sorted dict: {snr_tasks}')
        pub.sendMessage('snr', nii_path=self._instances[self._key]['nii_path'], tasks=snr_tasks)
        logger.debug('after snr calculation')

        snr_process = multiprocessing.Process(target=self.snr_controller, args=(ds,))
        snr_process.start()


        logger.debug(json.dumps(self._instances, indent=2))

        logger.debug(f'after volreg')
        logger.debug(json.dumps(self._instances, indent=2))
        project = ds.get('StudyDescription', '[STUDY]')
        session = ds.get('PatientID', '[PATIENT]')
        scandesc = ds.get('SeriesDescription', '[SERIES]')
        scannum = ds.get('SeriesNumber', '[NUMBER]')
        subtitle_string = f'{project} • {session} • {scandesc} • {scannum}'
        pub.sendMessage('plot', instances=self._instances, subtitle_string=subtitle_string)

        


    def snr_controller(self, ds):

            
            logger.info(f'received snr controller topic notification')

            key = int(ds.InstanceNumber)
            num_vols = ds[(0x0020, 0x0105)].value

            if key == 1:
                self._mask_threshold, self._decrement = self.get_mask_threshold(ds)
                x, y, z, _ = self._slice_means[key]['slice_means'].shape

                self._prev_mask = None
                self._slices_to_update = None



                self._z = z
                self._num_vols = num_vols

                self._slice_intensity_means = np.zeros( (z, num_vols) )

                self._fdata_array = np.empty((x, y, z, num_vols))

                logger.info(f'shape of zeros: {self._fdata_array.shape}')
                logger.info(f'shape of first slice means: {self._slice_means[key]['slice_means'].shape}')

            if key >= 5:
                insert_position = key - 5
                self._fdata_array[:, :, :, insert_position] = self._slice_means[key]['slice_means'].squeeze()

            if key > 53 and (key % 1 == 0) and key < num_vols:
                logger.info(f'shape of fdata_array: {self._fdata_array.shape}')
                logger.info('publishing message to plot_snr topic')

            #snr_thread = threading.Thread(target=self.calculate_and_publish_snr)
            #snr_thread.start()
            #snr_process.join()
                self.calculate_and_publish_snr(key)
            #snr_metric = round(self.calc_snr(), 2)
            #logger.info(f'running snr metric: {snr_metric}')
            #pub.sendMessage('plot_snr', snr_metric=snr_metric)

            if key == num_vols:
                logger.debug('RUNNING FINAL SNR CALCULATION')
                snr_metric = round(self.calc_snr(key), 2)
                logger.info(f'final snr metric: {snr_metric}')
                pub.sendMessage('plot_snr', snr_metric=snr_metric)

            '''
            matplotlib.use('Agg')
            x, y, z, _ = self._slice_means[self._key]['slice_means'].shape
            plt.figure(figsize=(10, 6))
            plt.plot(self._plot_dict.keys(), self._plot_dict.values(), marker='o', linestyle='-', color='b')
            plt.title(f"Dimensions: {x}x{y}x{z}")
            plt.xlabel("Volume Number")
            plt.ylabel("Processing Time")
            plt.savefig('/Users/danielasay/Desktop/buckner_proc_time.png')
            '''


    def calculate_and_publish_snr(self, key):
        start = time.time()
        snr_metric = round(self.calc_snr(key), 2)
        elapsed = time.time() - start
        #self._plot_dict[self._key] = elapsed
        logger.info(f'snr calculation took {elapsed} seconds')
        logger.info(f'running snr metric: {snr_metric}')
        if np.isnan(snr_metric):
            logger.info(f'snr is a nan, decrementing mask threshold by {self._decrement}')
            self._mask_threshold = self._mask_threshold - self._decrement
            logger.info(f'new threshold: {self._mask_threshold}')
            self._numpy_4d_mask = np.zeros(self._fdata_array.shape, dtype=bool)
            self._slice_intensity_means = np.zeros( (self._z, self._num_vols) )
        else:
            pub.sendMessage('plot_snr', snr_metric=snr_metric)

    def get_snr_interval(self, num_dicoms):
        if num_dicoms == 0:
            return max(self._snr_interval - 1, 2)
        if num_dicoms > 1:
            return self._snr_interval + 1


    def check_volreg(self, key):
        tasks = list()
        current = self._instances[key]

        # get numerical index of self._key O(log n)
        i = self._instances.bisect_left(key)

        # always register current node to left node
        try:
            left_index = max(0, i - 1)
            left = self._instances.values()[left_index]
            logger.debug(f'to the left of {current["path"]} is {left["path"]}')
            tasks.append((current, left))
        except IndexError:
            pass

        # if there is a right node, re-register to current node
        try:
            right_index = i + 1
            right = self._instances.values()[right_index]
            logger.debug(f'to the right of {current["path"]} is {right["path"]}')
            tasks.append((right, current))
        except IndexError:
            pass

        return tasks

    def calc_snr(self, key):

        slice_intensity_means, slice_voxel_counts, data = self.get_mean_slice_intensitites(key)



        #pdb.set_trace()

        non_zero_columns = ~np.all(slice_intensity_means == 0, axis=0)

        slice_intensity_means_2 = slice_intensity_means[:, non_zero_columns]

        #idx = np.argwhere(np.all(slice_intensity_means[..., :] == 0, axis=0))

        #slice_intensity_means_2 = np.delete(slice_intensity_means, idx, axis=1)

        slice_count = slice_intensity_means_2.shape[0]
        volume_count = slice_intensity_means_2.shape[1]

        slice_weighted_mean_mean = 0
        slice_weighted_stdev_mean = 0
        slice_weighted_snr_mean = 0
        slice_weighted_max_mean = 0
        slice_weighted_min_mean = 0
        outlier_count = 0
        total_voxel_count = 0

        for slice_idx in range(slice_count):
            slice_data         = slice_intensity_means_2[slice_idx]
            slice_voxel_count  = slice_voxel_counts[slice_idx]
            slice_mean         = slice_data.mean()
            slice_stdev        = slice_data.std(ddof=1)
            slice_snr          = slice_mean / slice_stdev

            slice_weighted_mean_mean   += (slice_mean * slice_voxel_count)
            slice_weighted_stdev_mean  += (slice_stdev * slice_voxel_count)
            slice_weighted_snr_mean    += (slice_snr * slice_voxel_count)

            total_voxel_count += slice_voxel_count

            logger.debug(f"Slice {slice_idx}: Mean={slice_mean}, StdDev={slice_stdev}, SNR={slice_snr}")
            
        #pdb.set_trace()

        return slice_weighted_snr_mean / total_voxel_count


    def get_mean_slice_intensitites(self, key):

        data = self.generate_mask(key)
        mask = np.ma.getmask(data)
        #dim_x, dim_y, dim_z, dim_t = data.shape
        dim_x, dim_y, dim_z, _ = data.shape

        dim_t = key - 4

        #slices_to_update = list()

        if key > 54:
            differing_slices = self.find_mask_differences(key)
            #pdb.set_trace()

        #home_dir = os.path.expanduser("~")
        #slice_intensity_means_read = np.load(f'{home_dir}/slice_intensity_means.npy')
        #slice_intensity_means = slice_intensity_means_read[:,:dim_t]
        slice_voxel_counts = np.zeros( (dim_z), dtype='uint32' )
        slice_size = dim_x * dim_y

        for slice_idx in range(dim_z):
            slice_voxel_counts[slice_idx] = slice_size - mask[:,:,slice_idx,0].sum()

        #pdb.set_trace()

        #column_sums = np.sum(array, axis=0)
        zero_columns = np.where(np.all(self._slice_intensity_means[:,:dim_t] == 0, axis=0))[0].tolist()

        logger.info(zero_columns)

        if len(zero_columns) > 20:
            for volume_idx in zero_columns:
                for slice_idx in range(dim_z):
                    slice_data = data[:,:,slice_idx,volume_idx]
                    self._slice_intensity_means[slice_idx,volume_idx] = slice_data.mean()
        else:

        #    pdb.set_trace()

            for volume_idx in zero_columns:
                for slice_idx in range(dim_z):
                    slice_data = data[:,:,slice_idx,volume_idx]
                    slice_vol_mean = slice_data.mean()
                    #whole_slice_mean = self._slice_intensity_means[:,volume_idx-1].mean()
                    #pdb.set_trace()
                    #weighted_mean = (slice_vol_mean + whole_slice_mean) / (volume_idx+1)
                    self._slice_intensity_means[slice_idx,volume_idx] = slice_vol_mean

            logger.info(f'recalculating slice means at the following slices: {differing_slices}')
            logger.info(f'total of {len(differing_slices)} new slices being computed')

            if differing_slices:

                for volume_idx in range(dim_t):
                    for slice_idx in differing_slices:
                        slice_data = data[:,:,slice_idx,volume_idx]
                        slice_vol_mean = slice_data.mean()
                        self._slice_intensity_means[slice_idx,volume_idx] = slice_vol_mean
                #self._slice_intensity_means[slice_idx,volume_idx] = slice_data.mean()

        

        #_, _, _, old_vols = self._prev_mask.shape

        #differences = self._prev_mask != mask[:,:,:,:old_vols]

        #if np.any(differences):
        #    logger.info('found differences!')
        #    pdb.set_trace()
        #    diff_indices = np.where(differences)
            #for index in zip(*diff_indices):
                #logger.info(f"Index: {index}, Previous volume data: {self._prev_mask[index]}, Current Data: {mask[index]}")
                #logger.info(f'differences between masks at volume {key}')



        #pdb.set_trace()


        #self._prev_mask = mask

        #self._slice_intensity_means_read[:,:dim_t] = slice_intensity_means

        return self._slice_intensity_means[:, :dim_t], slice_voxel_counts, data

        #np.save(f'{home_dir}/slice_intensity_means.npy', slice_intensity_means_read)

        #return slice_intensity_means, slice_voxel_counts, data

        # Update self._slice_intensity_means in the same manner
        #self._slice_intensity_means[:, volume_idx] = slice_data_sum / slice_voxel_counts

        '''

        slice_intensity_means = np.zeros( (dim_z,dim_t) )
        slice_voxel_counts = np.zeros( (dim_z), dtype='uint32' )
        slice_size = dim_x * dim_y

        for slice_idx in range(dim_z):
            slice_voxel_counts[slice_idx] = slice_size - mask[:,:,slice_idx,0].sum()

        
        for volume_idx in range(dim_t):
            for slice_idx in range(dim_z):
                slice_data = data[:,:,slice_idx,volume_idx]
                slice_intensity_means[slice_idx,volume_idx] = slice_data.mean()

        np.save(f'/Users/danielasay/Desktop/slice_intensity_means_{dim_t}_not_optimized.npy', slice_intensity_means)        

        #pdb.set_trace()

        if self._slice_intensity_means[30][30] == 0.0:

            for volume_idx in range(dim_t):
                for slice_idx in range(dim_z):
                    slice_data = data[:,:,slice_idx,volume_idx]
                    self._slice_intensity_means[slice_idx,volume_idx] = slice_data.mean().squeeze()
            #pdb.set_trace()
        else:
            for slice_idx in range(dim_z):
                slice_data = data[:,:,slice_idx,dim_t-1]
                slice_mean = slice_data.mean()
                #pdb.set_trace()
                self._slice_intensity_means[slice_idx, dim_t-1] = slice_mean

        #if key == 330:
        np.save(f'/Users/danielasay/Desktop/slice_intensity_means_{dim_t}_optimized.npy', self._slice_intensity_means[:,:dim_t])
        np.save(f'/Users/danielasay/Desktop/slice_intensity_means_{dim_t}_not_optimized.npy', slice_intensity_means)

        #pdb.set_trace()

        return self._slice_intensity_means[:, :dim_t], slice_voxel_counts, data 
        '''        


    def generate_mask(self, key):

        mean_data = np.mean(self._fdata_array[..., :key-4], axis=3)

        numpy_3d_mask = np.zeros(mean_data.shape, dtype=bool)

        to_mask = (mean_data <= self._mask_threshold)

        mask_lower_count = int(to_mask.sum())

        numpy_3d_mask = numpy_3d_mask | to_mask

        numpy_4d_mask = np.zeros(self._fdata_array[..., :self._key-4].shape, dtype=bool)

        numpy_4d_mask[numpy_3d_mask] = 1

        masked_data = np.ma.masked_array(self._fdata_array[..., :key-4], mask=numpy_4d_mask[..., :key-4])

        mask = np.ma.getmask(masked_data)

        self._slice_means[key]['mask'] = mask

        return masked_data

        #if self._prev_mask is None:
        #    self._prev_mask = np.ma.getmask(masked_data)

        #_, _, _, old_vols = self._prev_mask.shape

        #differences = self._prev_mask != mask[:,:,:,:old_vols]

        #if np.any(differences):
        #    logger.info('found differences!')
        #    pdb.set_trace()

        '''
        if key == 54:
            masked_data.dump('/Users/danielasay/50_vol_data.npy')
            mask = np.ma.getmask(masked_data)
            mask.dump('/Users/danielasay/50_vol_mask.npy')

        if key == 55:
            masked_data.dump('/Users/danielasay/51_vol_data.npy')
            mask = np.ma.getmask(masked_data)
            mask.dump('/Users/danielasay/51_vol_mask.npy')

            sys.exit()

        if key == 154:
            #np.dump('/Users/danielasay/150_vol_mask.npy', masked_data)
            masked_data.dump('/Users/danielasay/150_vol_data.npy')
            mask = np.ma.getmask(masked_data)
            mask.dump('/Users/danielasay/150_vol_mask.npy')
        '''

        #return masked_data

    def find_mask_differences(self, key):
        num_old_vols = key - 5 
        prev_mask = self._slice_means[key-1]['mask']
        current_mask = self._slice_means[key]['mask']
        differences = prev_mask != current_mask[:,:,:,:num_old_vols]
        diff_indices = np.where(differences)
        differing_slices = []
        for index in zip(*diff_indices):
            if int(index[2]) not in differing_slices:
                differing_slices.append(int(index[2]))
        slices_to_update = differing_slices.sort()
        #pdb.set_trace()
        return differing_slices



    def get_mask_threshold(self, ds):
        bits_stored = ds.get('BitsStored', None)
        receive_coil = self.find_coil(ds)

        if bits_stored == 12:
            logger.debug(f'scan has "{bits_stored}" bits and receive coil "{receive_coil}", setting mask threshold to 150.0')
            return 150.0, 10
        if bits_stored == 16:
            if receive_coil in ['Head_32']:
                logger.debug(f'scan has "{bits_stored}" bits and receive coil "{receive_coil}", setting mask threshold to 1500.0')
                return 1500.0, 100
            if receive_coil in ['Head_64', 'HeadNeck_64']:
                logger.debug(f'scan has "{bits_stored}" bits and receive coil "{receive_coil}", setting mask threshold to 3000.0')
                return 3000.0, 300
        raise MaskThresholdError(f'unexpected bits stored "{bits_stored}" + receive coil "{receive_coil}"')

    def find_coil(self, ds):
        seq = ds[(0x5200, 0x9229)][0]
        seq = seq[(0x0018, 0x9042)][0]
        return seq[(0x0018, 0x1250)].value


    def get_slice_voxel_count(self, lower_threshold_to_zero=None):
        """
        Compute the voxel counts per slice based on a masking array and optional thresholding.
        """ 

        # Apply mask to the data with given mask array
        #masked_data = np.ma.masked_array(self._slice_means_array[:, i], mask=self._mask_array[])    

        # Apply threshold if specified
        #if lower_threshold_to_zero is not None:
        #    masked_data = np.ma.where(masked_data > lower_threshold_to_zero, masked_data, 0)    

        # Number of slices (z-dimension)
        num_slices = self._slice_means_array.shape[0]
        
        # Initialize the array to hold voxel counts for each slice
        slice_voxel_counts = np.zeros(num_slices, dtype=int)    

        # Calculate voxels for each slice in the z-dimension
        for slice_idx in range(num_slices):
            # Counting unmasked (valid) voxels in each slice
            slice_voxel_counts[slice_idx] = np.sum(~self._mask_array[slice_idx])    

        return slice_voxel_counts


    def check_snr(self, key):
        tasks = list()
        current = self._slice_means[self._key]

        current_idx = self._slice_means.bisect_left(self._key)

        try:
            value = self._slice_means.values()[current_idx]
            tasks.append(value)
        except IndexError:
            pass

        return tasks


