#Annotation: Akira

# currdf is obtained via pd.read_csv(filename, low_memory=False)
# and looks like:
# column 0: native index
# column 1: index, 0 - (as many entries)
# -----
# row 0: body parts (snout, rightforepaw, leftforepaw,
#        righthindpaw, lefthindpaw, tailbase, belly)
# row 1: entries, (coords, [x, y, likelihood] * 7)
# row 2: position over time start


# The output of this function, curr_df_filt is:
# a [number of timepoints x (2 * number of pody parts)] numpy float array
# where each two columns are x and y positions from the same body position,
# where body positions are arranged according to the order of the "pose" variable
# passed, and the row entries are numerical values of their position after filtering
# of probabilities
def adp_filt(currdf: object, pose):
    lIndex = []
    xIndex = []
    yIndex = []
    # removing a header row that contains body parts
    currdf = np.array(currdf[1:])
    # ASSUMIGN THIS IS BODY PARTS + ATTRIBUTE (E.G. SNOUT LIKELIHOOD)
    for header in pose:
        if currdf[0][header + 1] == "likelihood":
            lIndex.append(header)
        elif currdf[0][header + 1] == "x":
            xIndex.append(header)
        elif currdf[0][header + 1] == "y":
            yIndex.append(header)
    # removing the first column, which are the indices
    curr_df1 = currdf[:, 1:]
    # getting every x, y and likelihood columns into their own data frames
    # these dataframes only hold numerical values
    datax = curr_df1[1:, np.array(xIndex)]
    datay = curr_df1[1:, np.array(yIndex)]
    data_lh = curr_df1[1:, np.array(lIndex)]
    # create a numpy zeros used to store x and y data at once?
    # datax.shape[0] and [1] are the number of rows and columns
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = [0] * data_lh.shape[1]
    # for the number of columns in data_lh (corresponding to how many body parts there are)
    for curr_bodypart_idx in tqdm(range(data_lh.shape[1])):
        # we convert the data array into a histogram, getting bins
        # number of bins defaults to 10
        count_in_bins, bin_edges = np.histogram(data_lh[1:, curr_bodypart_idx].astype(np.float))
        # we find where the count in bins increases
        rise_a = np.where(np.diff(count_in_bins) >= 0)
        if rise_a[0][0] > 1:
            llh = bin_edges[rise_a[0][0]]
        else:
            llh = bin_edges[rise_a[0][1]]
        # seems like we are setting llh anyway, so it doesn't matter
        ######################
        # THIS MIGHT BE ELLEN'S ADDITION, WHICH SET THE CUTOFF & PREDICTION LIMIT AUTOMATICALLY TO 0.8
        ######################
        llh = 0.8
        # get the specific likelihood column as float np array
        data_lh_float = data_lh[:, curr_bodypart_idx].astype(np.float)
        # we then compute the percentage of entries that were rectified (because the probability went below llh)
        # for this specific body  part
        perc_rect[curr_bodypart_idx] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        # our new dataframe holds the ith body part x & y data at (2i) and (2i + 1)
        new_x_col_idx, new_y_col_idx = (2 * curr_bodypart_idx), (2 * curr_bodypart_idx + 1)
        # we copy the 0th row's entry to begin with 
        currdf_filt[0, new_x_col_idx] = datax[0, curr_bodypart_idx]
        currdf_filt[0, new_y_col_idx] = datay[0, curr_bodypart_idx]
        # for every remainig row in data likelihood
        for curr_row_idx in range(1, data_lh.shape[0]):
            # if this row's entry is smaller than the cutoff, we use the previous position prediction
            if data_lh_float[curr_row_idx] < llh:
                currdf_filt[curr_row_idx, new_x_col_idx:new_y_col_idx+1] = currdf_filt[curr_row_idx - 1, new_x_col_idx:new_y_col_idx+1]
            # otherwise, we normally copy the position prediction into the new dataframe
            else:
                currdf_filt[curr_row_idx, new_x_col_idx:new_y_col_idx+1] = np.hstack(
                    [datax[curr_row_idx, curr_bodypart_idx], datay[curr_row_idx, curr_bodypart_idx]]
                    )
    # cast the pd frame into a float numpy array
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect