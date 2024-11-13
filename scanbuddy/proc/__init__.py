import os
import sys
import pdb
import time
import math
import json
import psutil
import logging
import threading
import matplotlib
import numpy as np
from pubsub import pub
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)

class Processor:
    def __init__(self):
        self.reset()
        pub.subscribe(self.reset, 'reset')
        pub.subscribe(self.listener, 'incoming')

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
            'mask_threshold': None
        }
        logger.debug('current state of instances')
        logger.debug(json.dumps(self._instances, default=list, indent=2))

        num_vols = ds[(0x0020, 0x0105)].value

        #if self._key == 1:

            #self._plot_dict = dict.fromkeys(range(1, num_vols+1), None)

        data_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))

        #logger.info(f'data path {data_path}')

        num_dicoms = len([name for name in os.listdir(data_path) if name.endswith('.dcm')])

        if num_dicoms > 0:
            self._snr_interval = 100
        else:
            self._snr_interval = 5

#        interval_mapping = {
#            0: 2,
#            1: 3,
#            2: 4,
#            3: 5,
#            4: 6,
#            5: 7,
#            6: 8
#        }  
#            if self._key == 1:
#                self._snr_interval = 1
#            else:
#                self._snr_interval = self.get_snr_interval(num_dicoms)
        #if num_dicoms > 1:
            #self._snr_interval     

        #self._snr_interval = interval_mapping.get(num_dicoms, 9)

        logger.info(f'CURRENT SNR INTERVAL {self._snr_interval}')


        #self._plot_dict[self._key] = num_dicoms

        #self._plot_dict[self._key] = psutil.cpu_percent()

        #pdb.set_trace()

        tasks = self.check_volreg(self._key)
        ### edits need to be made here
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

        logger.debug(json.dumps(self._instances, indent=2))



        if self._key == 1:
            self._mask_threshold = self.get_mask_threshold(ds)
            x, y, z, _ = self._slice_means[self._key]['slice_means'].shape

            #self._fdata_array = np.zeros((x, y, z, num_vols))
            self._fdata_array = np.empty((x, y, z, num_vols))
            self._numpy_4d_mask = np.zeros(self._fdata_array.shape, dtype=bool)

            logger.info(f'shape of zeros: {self._fdata_array.shape}')
            logger.info(f'shape of first slice means: {self._slice_means[self._key]['slice_means'].shape}')

        if self._key >= 5:
            #pdb.set_trace()
            insert_position = self._key - 5
            #self._fdata_array[:, :, :, insert_position] = self._slice_means[self._key]['slice_means']
            self._fdata_array[:, :, :, insert_position] = self._slice_means[self._key]['slice_means'].squeeze()

        #elif self._key > 5:
            #self._fdata_array = np.concatenate((self._fdata_array, self._slice_means[self._key]['slice_means']), axis=3)
        if self._key > 53 and (self._key % self._snr_interval == 0) and self._key < num_vols:
            #logger.info(f'shape of slice_means_array: {self._slice_means_array.shape}')
            logger.info(f'shape of fdata_array: {self._fdata_array.shape}')
            logger.info('publishing message to plot_snr topic')

            snr_thread = threading.Thread(target=self.calculate_and_publish_snr)
            snr_thread.start()
            #snr_metric = round(self.calc_snr(), 2)
            #logger.info(f'running snr metric: {snr_metric}')
            #pub.sendMessage('plot_snr', snr_metric=snr_metric)

        if self._key == num_vols:
            #snr_thread.stop()
            logger.debug('RUNNING FINAL SNR CALCULATION')
            snr_metric = round(self.calc_snr(), 2)
            logger.info(f'running snr metric: {snr_metric}')
            pub.sendMessage('plot_snr', snr_metric=snr_metric)

            #matplotlib.use('Agg')

            #x, y, z, _ = self._slice_means[self._key]['slice_means'].shape

            #plt.figure(figsize=(10, 6))
            #plt.plot(self._plot_dict.keys(), self._plot_dict.values(), marker='o', linestyle='-', color='b')
            #plt.title(f"Dimensions: {x}x{y}x{z}")
            #plt.xlabel("Volume Number")
            #plt.ylabel("Processing Time")

            #plt.savefig('/Users/danielasay/Desktop/star_num_dicoms.png')



            
        logger.debug(f'after volreg')
        logger.debug(json.dumps(self._instances, indent=2))
        project = ds.get('StudyDescription', '[STUDY]')
        session = ds.get('PatientID', '[PATIENT]')
        scandesc = ds.get('SeriesDescription', '[SERIES]')
        scannum = ds.get('SeriesNumber', '[NUMBER]')
        subtitle_string = f'{project} • {session} • {scandesc} • {scannum}'
        pub.sendMessage('plot', instances=self._instances, subtitle_string=subtitle_string)

    def calculate_and_publish_snr(self):
        start = time.time()
        snr_metric = round(self.calc_snr(), 2)
        elapsed = time.time() - start
        #self._plot_dict[self._key] = elapsed
        logger.info(f'snr calculation took {elapsed} seconds')
        logger.info(f'running snr metric: {snr_metric}')
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

    def calc_snr(self):

        slice_intensity_means, slice_voxel_counts, data = self.get_mean_slice_intensitites()

        slice_count = slice_intensity_means.shape[0]
        volume_count = slice_intensity_means.shape[1]

        slice_weighted_mean_mean = 0
        slice_weighted_stdev_mean = 0
        slice_weighted_snr_mean = 0
        slice_weighted_max_mean = 0
        slice_weighted_min_mean = 0
        outlier_count = 0
        total_voxel_count = 0

        for slice_idx in range(slice_count):
            slice_data         = slice_intensity_means[slice_idx]
            slice_voxel_count  = slice_voxel_counts[slice_idx]
            slice_mean         = slice_data.mean()
            slice_stdev        = slice_data.std(ddof=1)
            slice_snr          = slice_mean / slice_stdev

            slice_weighted_mean_mean   += (slice_mean * slice_voxel_count)
            slice_weighted_stdev_mean  += (slice_stdev * slice_voxel_count)
            slice_weighted_snr_mean    += (slice_snr * slice_voxel_count)

            total_voxel_count += slice_voxel_count

        #return slice_weighted_mean_mean / total_voxel_count
        return slice_weighted_snr_mean / total_voxel_count


    def get_mean_slice_intensitites(self):

        data = self.generate_mask()
        mask = np.ma.getmask(data)
        #dim_x, dim_y, dim_z, dim_t = data.shape
        dim_x, dim_y, dim_z, _ = data.shape

        dim_t = self._key - 4

        slice_intensity_means = np.zeros( (dim_z,dim_t) )
        slice_voxel_counts = np.zeros( (dim_z), dtype='uint32' )
        slice_size = dim_x * dim_y

        for slice_idx in range(dim_z):
            slice_voxel_counts[slice_idx] = slice_size - mask[:,:,slice_idx,0].sum()

        for volume_idx in range(dim_t):
            for slice_idx in range(dim_z):
                slice_data = data[:,:,slice_idx,volume_idx]
                slice_intensity_means[slice_idx,volume_idx] = slice_data.mean()

        return slice_intensity_means, slice_voxel_counts, data


    def generate_mask(self):

        mean_data = np.mean(self._fdata_array[..., :self._key-4], axis=3)

        #pdb.set_trace()

        numpy_3d_mask = np.zeros(mean_data.shape, dtype=bool)

        to_mask = (mean_data <= self._mask_threshold)

        mask_lower_count = int(to_mask.sum())

        numpy_3d_mask = numpy_3d_mask | to_mask

        #numpy_4d_mask = np.zeros(self._fdata_array[..., :self._key-4].shape, dtype=bool)

        self._numpy_4d_mask[numpy_3d_mask] = 1

        masked_data = np.ma.masked_array(self._fdata_array[..., :self._key-4], mask=self._numpy_4d_mask[..., :self._key-4])

        return masked_data


    def get_mask_threshold(self, ds):
        bits_stored = ds.get('BitsStored', None)
        receive_coil = self.find_coil(ds)

        if bits_stored == 12:
            logger.debug(f'scan has "{bits_stored}" bits and receive coil "{receive_coil}", setting mask threshold to 150.0')
            return 150.0
        if bits_stored == 16:
            if receive_coil in ['Head_32']:
                logger.debug(f'scan has "{bits_stored}" bits and receive coil "{receive_coil}", setting mask threshold to 1500.0')
                return 1500.0
            if receive_coil in ['Head_64', 'HeadNeck_64']:
                logger.debug(f'scan has "{bits_stored}" bits and receive coil "{receive_coil}", setting mask threshold to 3000.0')
                return 3000.0
        raise MaskThresholdError(f'unexpected bits stored "{bits_stored}" + receive coil "{receive_coil}"')

    def find_coil(self, ds):
        seq = ds[(0x5200, 0x9229)][0]
        seq = seq[(0x0018, 0x9042)][0]
        return seq[(0x0018, 0x1250)].value


        #pdb.set_trace()

        '''

        if self._slice_means_array.shape[0] != self._mask_array.shape[2]:
            raise ValueError("The number of slices in mean_voxel_intensities must equal the number of slices in binary_mask.")


        slice_weighted_mean_mean = 0.0
        slice_weighted_stdev_mean = 0.0
        total_voxel_count = 0

        for slice_idx in range(self._slice_means_array.shape[0]):
            slice_voxel_count = np.sum(self._mask_array[:,:,slice_idx] == False)
            slice_data = self._slice_means_array[slice_idx, :]

            if slice_voxel_count > 0:
                slice_mean = np.mean(slice_data)
                slice_stdev = np.std(slice_data, ddof=1)

                if np.isfinite(slice_mean) and np.isfinite(slice_stdev):
                    slice_weighted_mean_mean += (slice_mean * slice_voxel_count)
                    slice_weighted_stdev_mean += (slice_stdev * slice_voxel_count)
                    total_voxel_count += slice_voxel_count

        if total_voxel_count == 0:
            raise ValueError("Total voxel count is zero. Check your binary mask.")  

        slice_weighted_mean = slice_weighted_mean_mean / total_voxel_count
        slice_weighted_stdev = slice_weighted_stdev_mean / total_voxel_count    

        slice_weighted_snr = slice_weighted_mean / slice_weighted_stdev if slice_weighted_stdev != 0 else 0 

        return slice_weighted_snr
        '''

        '''
        slice_wmean_sum = 0
        slice_wstd_sum = 0
        slice_wsnr_sum = 0
        total_masked_voxels = 0
        total_voxel_count = 0
        slice_summary = []

        slice_voxel_counts = self.get_slice_voxel_count()

        for slice_idx in range(self._slice_means_array.shape[0]):

            volume_means = self._slice_means_array[slice_idx, :]
            # compute the raw slice mean across time
            mean = volume_means.mean()
            # compute the raw slice std across time
            std = volume_means.std(ddof=1)
            if std == 0:
                snr = 0
            else:
                snr = mean / std

            # store raw statistics within the slice summary
            #slice_summary.append(SliceRow(i + 1, mean, std, snr))
            # count the number of un-masked voxels for the i'th slice

            #num_masked_voxels = (self._mask_array[:,:,slice_idx] == True).sum()

            slice_voxel_count = slice_voxel_counts[slice_idx]

            slice_wsnr_sum += snr * slice_voxel_count

            total_voxel_count += slice_voxel_count

            # keep a running sum of the number of un-masked voxels

            #total_masked_voxels += num_masked_voxels

            # keep running sums of the weighted mean, std, and snr
            #slice_wmean_sum += mean * num_masked_voxels
            #slice_wstd_sum += std * num_masked_voxels

            #slice_wsnr_sum += snr * num_masked_voxels

        # compute the weighted slice-based mean, standard deviation, and snr
        #wm = slice_wmean_sum / total_masked_voxels
        #ws = slice_wstd_sum / total_masked_voxels
        wsnr = slice_wsnr_sum / total_voxel_count

        logger.debug(wsnr)
        
        return wsnr
        '''

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
            #pdb.set_trace()
            #masked_data = np.ma.masked_array(self._slice_means_array[slice_idx,:], mask=self._mask_array[:,:,slice_idx])
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

        ## get the index thats snr_interval (10, for example) to the left of the current
        '''
        try:
            left_index = current_idx - int(self._snr_interval)
            for idx in range(left_index, current_idx):
                value = self._instances.values()[idx]
                logger.debug(f'adding dicom {value["path"]} to snr calculation task')
                tasks.append(value)
        except IndexError:
            pass

        return tasks

        # extract the latest snr value calculation
        
        if len(self._instances) == self._snr_interval:
            prev_snr = 0.0
        else:
            for self._key in reversed(self._instances.self._keys()):
                if self._instances[self._key]['snr'] is None:
                    continue
                else:
                    prev_snr = float(self._instances[self._key]['snr'])
                    logger.info(f'found previously calculated snr at index {self._key}')
                    return tasks, prev_snr
        '''

