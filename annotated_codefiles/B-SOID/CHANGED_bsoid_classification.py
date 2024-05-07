import itertools
import math

import numpy as np

from bsoid_app.bsoid_utilities.likelihoodprocessing import boxcar_center


def bsoid_extract(data, fps):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
    """
    # length of window after this;
    # e.g. for a 40 fps video, this is:
    # 0.05 / (0.025) * 2 - 1 = 4 - 1 = 3
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    # for each csv data - format of CSV data can be found under CHANGED_likelihoodprocessing.py
    for csv_data in data:
        num_datapoint = len(csv_data)
        dxy_r = [] #estimated: [21 x timepoints]
        dis_r = [] #estimated: [7 x timepoints]
        # for each time point in the data
        for curr_timepoint in range(num_datapoint):
            # we only look at datapoints before the final one
            if curr_timepoint < num_datapoint - 1:
                dis = []
                # for every x column (we have an x column followed by a y column in the dataset)
                # in our case we have 7 body parts
                for c in range(0, csv_data.shape[1], 2):
                    # calculate the distance between consecutive points in time
                    dis.append(np.linalg.norm(
                        csv_data[curr_timepoint + 1, c:c + 2] - # data for x and y position are in c:c+2 columns
                        csv_data[curr_timepoint,     c:c + 2]))
                dis_r.append(dis)
            dxy = []
            # itertools.combinations produces all possible combinations of x column number
            # in this case, we have 7 body parts for a total of 7C2 or 7*6/2/1 = 21 combinations
            for i, j in itertools.combinations(range(0, csv_data.shape[1], 2), 2):
                # this computes the distance between two body parts in the same frame
                dxy.append(csv_data[curr_timepoint, i:i + 2] - 
                           csv_data[curr_timepoint, j:j + 2])
            dxy_r.append(dxy)
        dis_r = np.array(dis_r)
        dxy_r = np.array(dxy_r)
        dis_smth = [] #estimated: [7 x timepoints]
        # we prepopulate two features, which will each have as many columns as combinations of 2 body parts
        # in this case we have 7C2 = 21
        dxy_eu = np.zeros([num_datapoint, dxy_r.shape[1]]) 
        # angle likely has one less because it is a second order computation?
        ang = np.zeros([num_datapoint - 1, dxy_r.shape[1]])
        dxy_smth = [] #estimated: [21 x timepoints]
        ang_smth = [] #estimated: [21 x timepoints]
        # dis_r has as many columns as the number of body parts (i.e. 7 for us)
        for curr_bodypart_idx in range(dis_r.shape[1]):
            # this smoothes the function using a "rolling window" approach, without reducing size of series
            dis_smth.append(boxcar_center(dis_r[:, curr_bodypart_idx], win_len))
        # dxy_r has as many columns as NC2 where N is the number of body parts (i.e. 21 for us)
        for curr_bodypartpair_idx in range(dxy_r.shape[1]):
            for curr_time_idx in range(num_datapoint):
                #UNSURE!  further reduction into Euclidean?
                dxy_eu[curr_time_idx, curr_bodypartpair_idx] = np.linalg.norm(dxy_r[curr_time_idx, curr_bodypartpair_idx, :])
                # so far as we don't go beyond the limits of the series
                if curr_time_idx < num_datapoint - 1:
                    # horizontally stack the dxy entry for the next timepoint and body part pair with 0
                    # UNSURE! this is how this body part moves at the next time step, in 3D?
                    b_3d = np.hstack([dxy_r[curr_time_idx + 1, curr_bodypartpair_idx, :], 0])
                    # horizontally stack the dxy entry for this timepoint and body part pair with 0
                    # UNSURE! this is how this body part moved at this time step, in 3D?
                    a_3d = np.hstack([dxy_r[curr_time_idx, curr_bodypartpair_idx, :], 0])
                    # compute the cross product of the two
                    c = np.cross(b_3d, a_3d)
                    # angle at this timepoint for this body part is computed from this, by the below (didn't go too deep)
                    ang[curr_time_idx, curr_bodypartpair_idx] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                        math.atan2(np.linalg.norm(c),
                                                   np.dot(dxy_r[curr_time_idx, curr_bodypartpair_idx, :], dxy_r[curr_time_idx + 1, curr_bodypartpair_idx, :])))
            # then smooth both results without changing series length
            dxy_smth.append(boxcar_center(dxy_eu[:, curr_bodypartpair_idx], win_len))
            ang_smth.append(boxcar_center(ang[:, curr_bodypartpair_idx], win_len))
        # turn them into np arrays
        dis_smth = np.array(dis_smth)
        dxy_smth = np.array(dxy_smth)
        ang_smth = np.array(ang_smth)
        # at this point, all features are [dim x timepoints] numpy arrays
        # append to the list of features a vertical stack of: 
        # 1) the smoothed distance between 21 body part pairs, all timepoints but the first
        # 2) the smoothed angle changes between 21 body pairs, all timepoints
        # 3) the consecutive distance changes for 7 body parts over time, all timepoints
        feats.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))
    f_10fps = []
    # then for every csv data we read
    for csv_idx, curr_feat in enumerate(feats):
        # create a numpy to populate with the same length as the length of the current feature
        feats1 = np.zeros(len(curr_feat))
        # for as many as we can create frameshifts given an fps higher than 10
        shift_num = round(fps / 10)
        for whole_shift_idx in range(shift_num):
            # from first shifted point all the way to the end of the end of time series, every
            # shift num points (essentially every small shifts)
            # e.g. with 40 fps, we are looking at every 4 points in the time series
            for n in range(shift_num + whole_shift_idx, len(curr_feat[0]), shift_num):
                # precompute where are the dxy_smth, ang_smth and dis_smth entries are as part of curr_feat
                curr_dxy_smth = curr_feat[0:dxy_smth.shape[0]]
                curr_ang_smth_and_dis_smth = curr_feat[dxy_smth.shape[0]:curr_feat.shape[0]]
                
                # for every timepoint but the first timepoint
                if n > shift_num + whole_shift_idx:
                    # feature is obtained by concatenating along axis 1 (the column axis, or time axis)
                    # 1 - the feats1 created so far
                    # 2 - a horizontal stack, [42 x 1] of (actual numbers depend on body parts we check):
                    #  2.1 - the mean of the distance between body part pairs over the range; [21 x 1]
                    #  2.2 - the total of the angular change between body part pairs over the range; [21 x 1]
                    #  2.3 - the total of the distance change between body part pairs over the range; [7 x 1]
                    feats1 = np.concatenate((
                        feats1.reshape(feats1.shape[0], feats1.shape[1]), # feat1, but reshaping to ensure the shape?
                        # a horizontal stack of:
                        np.hstack(
                            # the mean of this feature over this frameshift range
                            (np.mean((curr_dxy_smth[range(n - shift_num, n)]), axis=1),
                            # the sum of this feature over this frameshift range
                              np.sum((curr_ang_smth_and_dis_smth[range(n - shift_num, n)]), axis=1)
                             )
                            # then reshape the result into [row number of original feature vector x 1]
                             ).reshape(len(curr_feat), 1)
                            # finally append result at the end
                             ), axis=1)
                else:
                    # initial feature is a concatenation along axis 1 (the column axis, or time axis) of
                    # the above, except there is no prior feature vector
                    # CHECK OUT ABOVE FOR DETAILS!
                    feats1 = np.hstack((
                        np.mean((curr_dxy_smth[range(n - shift_num, n)]), axis=1),
                         np.sum((curr_ang_smth_and_dis_smth[range(n - shift_num, n)]), axis=1)
                         )).reshape(len(curr_feat), 1)
            # this downsampled frameshifted prediction is appended to f_10fps and returned
            f_10fps.append(feats1)
    return f_10fps


def bsoid_predict(feats, clf):
    """
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(0, len(feats)):
        labels = clf.predict(feats[i].T)
        labels_fslow.append(labels)
    return labels_fslow



