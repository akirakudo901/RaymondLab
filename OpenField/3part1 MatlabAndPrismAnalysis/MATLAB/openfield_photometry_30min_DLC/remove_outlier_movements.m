clear
close all

% trial removing outliers from the movement positions

%% Useful constants
PLOT_SPEED_DISTRIBUTION_BEFORE_AND_AFTER_FILTERING = false;
PLOT_POSITIONS_BEFORE_AND_AFTER_FILTERING = true;
THRESHOLD = 0.95;

%% Sanity check
% do run the sanity check functions
sanity_check_change_in_original_array(false)
sanity_check_filter_speed_by_upperbound()

%% Read and set up data
DATA_DIR = "data\temp_for_tests\" + ...
    "20230113142714_392607_m1_openfieldDLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv";    
% "312153_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv";
    

returned = setUp(DATA_DIR, THRESHOLD);
TailBaseX = returned.TailBase.X; TailBaseY = returned.TailBase.Y;

% FILTER BASED ON SPEED
% render_every_other_segment_with_different_color(TailBaseX, TailBaseY, "green", "magenta");
% [speedFilteredX, speedFilteredY] = filter_by_movement_outlier(TailBaseX, TailBaseY);
[speedFilteredX, speedFilteredY] = filter_by_speed_upperbound(TailBaseX, TailBaseY, 150);
% render_every_other_segment_with_different_color(speedFilteredX, speedFilteredY, "red", "blue");
% calculate difference in total distance traveled before & after filtering
reduction_in_distance(TailBaseX, TailBaseY, speedFilteredX, speedFilteredY);

% FILTER BASED ON POSITIONS OF BODY PARTS
% render_every_other_segment_with_different_color(speedFilteredX, speedFilteredY, "green", "magenta");
% cell array of body parts with struct
inputToMotionFilter = {
    struct('X', returned.Snout.X,         'Y', returned.Snout.Y),         ...
    struct('X', returned.RightFrontPaw.X, 'Y', returned.RightFrontPaw.Y), ...
    struct('X', returned.LeftFrontPaw.X,  'Y', returned.LeftFrontPaw.Y),  ...
    struct('X', returned.RightHindPaw.X,  'Y', returned.RightHindPaw.Y),  ...
    struct('X', returned.LeftHindPaw.X,   'Y', returned.LeftHindPaw.Y),   ...
    struct('X', speedFilteredX,           'Y', speedFilteredY),           ...
    struct('X', returned.Belly.X,         'Y', returned.Belly.Y),         ...
    };
motionFiltered = filter_by_body_part_position(inputToMotionFilter);
motionFilteredX = motionFiltered{6}.X; % 6 is the index of TailBase
motionFilteredY = motionFiltered{6}.Y; %  in inputToMotionFilter

% render_every_other_segment_with_different_color(motionFilteredX, motionFilteredY, "yellow", "black");
% calculate difference in total distance traveled before & after filtering
reduction_in_distance(speedFilteredX, speedFilteredY, motionFilteredX, motionFilteredY);



% we can also plot the speed distribution before and after filtering
if PLOT_SPEED_DISTRIBUTION_BEFORE_AND_AFTER_FILTERING
    originalSpeedPerFrame = sqrt(diff(TailBaseX).^2 + diff(TailBaseY).^2);
    newSpeedPerFrame      = sqrt(diff(motionFilteredX).^2 + diff(motionFilteredY).^2);

    figure;
    histogram(originalSpeedPerFrame);
    title("Distribution of speeds before filtering outliers");
    set(gca, 'yscale', 'log');

    figure;
    histogram(newSpeedPerFrame);
    title("Distribution of speeds after filtering outliers");
    set(gca, 'yscale', 'log');
end

% we plot the result
if PLOT_POSITIONS_BEFORE_AND_AFTER_FILTERING
    % first for speed filtering
    plot_difference_between_two_arrays(TailBaseX, TailBaseY, ...
        speedFilteredX, speedFilteredY, "Speed Filtered Tail Base");
    % then by position
    plot_difference_between_two_arrays(speedFilteredX, speedFilteredY, ...
        motionFilteredX, motionFilteredY, "Motion Filtered Tail Base");
end






%% Definition of helper functions
function [filteredX, filteredY] = filter_by_movement_outlier(X, Y)
    diffX = diff(X); diffY = diff(Y);
    totaldiff = sqrt(diffX.^2 + diffY.^2);

    % find all movement sequences between consecutive frames that is 
    % bigger than MOVE_THRESHOLD pixels - which we can think of 
    % as "movement"
    MOVE_THRESHOLD = 2;
    amovediff = totaldiff(totaldiff > MOVE_THRESHOLD); %if movement is greater than MOVE_THRESHOLD
    
    % then find the outliers for those movements in size
    omovediff = amovediff(isoutlier(amovediff,'mean'));
    
    % find the minimum such outlier value
    minout = min(omovediff);

    % filter by this upper bound
    [filteredX, filteredY] = filter_by_speed_upperbound(X, Y, minout);
end

% filters positions based on whether movements between them exceed a
% certain minSpeed per frame - and interpolate such positions
function [filteredX, filteredY] = filter_by_speed_upperbound(X, Y, ...
    upperBound)
    speedPerFrame = sqrt(diff(X).^2 + diff(Y).^2);
    % 1 - remove every entry in position for which movement leading to it
    %     was faster than the upper bound given
    % 2 - for every position after a removed position, compute distance
    %     between it and the last value that isn't removed
    % 3 - compute movement speed between those two positions, normalized by
    %     the number of frames elapsed for that movement
    % 4 - if this movement is still greater than the upper bound, remove it
    % 5 - repeat until reaching the end of the array

    % get values that exceed the upper bound movements
    aboveUpperBoundIndices = (speedPerFrame > upperBound);
    positionToBeRemoved = zeros(1, length(aboveUpperBoundIndices) + 1);
    positionToBeRemoved(2:end) = aboveUpperBoundIndices;
    % for each position in the arrays:
    % we cannot unfortunately exclude outlier movements happening at the
    % last time step
    metSpeedExceedingUpperBound = false;
    lastPositionX = 0; lastPositionY = 0;
    howManyPositionsWeSkipped = 0;
    
    for idx = 2:length(X)-1
        % if we are currently adjusting (we had an outlier before):
        if metSpeedExceedingUpperBound
            % compute the distance between the current posotion and the
            % previous position
            currPosX = X(idx); currPosY = Y(idx);
            deltaX = currPosX - lastPositionX;
            deltaY = currPosY - lastPositionY;
            distanceToPreviousPosition = sqrt(deltaX.^2 + deltaY.^2);
            % normalize it by the number of position we've skipped
            normDistanceOfMovement = distanceToPreviousPosition / howManyPositionsWeSkipped;
            % if this value is greater than the minimum of outlier:
            if normDistanceOfMovement > upperBound
                % we keep skipping: keep metSpeedExceedingUpperBound to 
                % true, and leave last position
                % but add 1 to how many positions we've skipped
                howManyPositionsWeSkipped = howManyPositionsWeSkipped + 1;
                % set the corresponding entry in the position array to NaN 
                % to extrapolate later
                X(idx) = NaN; Y(idx) = NaN;
            % otherwise, we got a valid distance
            else
                % we reset values and go back to checking mode
                metSpeedExceedingUpperBound = false; 
                lastPositionX = currPosX; lastPositionY = currPosY;
                howManyPositionsWeSkipped = 0;
            end
        else 
            % otherwise, we aren't adjusting, and will encouter either an
            % outlier or a normal one
            % if a movement is an outlier (by checking the index in the outlier
            % indices), we can:
            if positionToBeRemoved(idx)
                % set the corresponding entry in the position array to NaN to
                % extrapolate later
                X(idx) = NaN; Y(idx) = NaN;
                % we start adjusting
                metSpeedExceedingUpperBound = true;
                % we would increment the number of positions we've skipped
                howManyPositionsWeSkipped = howManyPositionsWeSkipped + 1;
            % otherwise, we got a valid distance
            else
                % we keep checking 
                lastPositionX = X(idx); 
                lastPositionY = Y(idx);
                howManyPositionsWeSkipped = 0;
            end
        end
    end
    
    % finally, we interpolate the remaining
    filteredX = repnan(X);
    filteredY = repnan(Y);
end

% Filters and extrapolates body parts position, based on whether the body
% part is an outlier considering the distance between it and the average 
% of all other body parts on that frame.
% Parameter is a cell array where each cell is a struct holding 
% the X and Y positions of individual body parts.
% Returns a struct holding the filtered body part arrays in the same 
% order as they were passed.
function filtered = filter_by_body_part_position(XYCellArray)
    if ~iscell(XYCellArray) 
        return; 
    end
    % get array to populate with average X and Y
    averageXY = zeros([length(XYCellArray{1}(1).X), 2]);
    
    % for each entry in the cell array
    for idx = 1:length(XYCellArray)
        % add up the x and y coordinates
        xyStruct = XYCellArray{idx}(1);
        averageXY(:,1) = averageXY(:,1) + xyStruct.X;
        averageXY(:,2) = averageXY(:,2) + xyStruct.Y;
    end
    % get the average
    averageXY = averageXY / length(XYCellArray);
    % for every entry in XY arrays of body positions, get outliers on
    % distance from the average XY position
    filtered = cell(size(XYCellArray));
    
    for idx = 1:length(XYCellArray)
        filtered{idx} = struct( ...
            'X', XYCellArray{idx}.X, 'Y', XYCellArray{idx}.Y);

        xyStruct = XYCellArray{idx}(1);
        % get outliers indices and populate with NaN
        distanceFromAverageX = abs(xyStruct.X - averageXY(1));
        distanceFromAverageY = abs(xyStruct.Y - averageXY(1));
        outlierX = isoutlier(distanceFromAverageX,'mean');
        outlierY = isoutlier(distanceFromAverageY,'mean');
        filtered{idx}.X(outlierX) = NaN;
        filtered{idx}.Y(outlierY) = NaN;
        % interpolate using linear interpolation
        filtered{idx}.X(outlierX) = fillmissing(filtered{idx}.X(outlierX), 'linear');
        filtered{idx}.Y(outlierY) = fillmissing(filtered{idx}.Y(outlierY), 'linear');
    end
end

%% Setup related
% Reads the data matrix and returns a structure of XY coordinate arrays 
% for different body parts
% Each body part array pairs can be accessed via name fields of those parts
function returned = setUp(dataDir, THRESHOLD)
    returned = struct();
    
    data = readmatrix(dataDir);
    % for each body part, X/Y/likelihood indices in csv
    bodyPartsAndIndices = struct( ...
        'Snout', [2,3,4], ...
        'RightFrontPaw', [5,6,7], ...
        'LeftFrontPaw', [8,9,10], ...
        'RightHindPaw', [11,12,13], ...
        'LeftHindPaw', [14,15,16], ...
        'TailBase', [17,18,19], ...
        'Belly', [20,21,22] ...
        );

    allFieldNames = fieldnames(bodyPartsAndIndices);
    for bodyPart = allFieldNames'
        bodyPartName = bodyPart{1};
        indices = bodyPartsAndIndices.(bodyPartName);
        [X, Y] = computeinterpolatedxycoordinates_withthreshold( ...
            indices(1), indices(2), indices(3), data, THRESHOLD);
        returned.(bodyPartName) = struct('X', X, 'Y', Y);
    end
end

% Produces arrays of x & y coordinates stored in xIdx and yIdx
% If corresponding likelihood (stored in lkhdIdx) of a coordinate is 
% less than THRESHOLD, interpolate it instead using the repnan function
function [xArray, yArray] = computeinterpolatedxycoordinates( ...
    xIdx, yIdx, lkhdIdx, data ...
    )
    % threshold of likelihood to interpolate given point
    THRESHOLD = 0.8;
    [xArray, yArray] = computeinterpolatedxycoordinates_withthreshold(xIdx, ...
        yIdx, lkhdIdx, data, THRESHOLD);
end

function [xArray, yArray] = computeinterpolatedxycoordinates_withthreshold( ...
    xIdx, yIdx, lkhdIdx, data, threshold...
    )
    xArray = data(:,xIdx);
    yArray = data(:,yIdx);
    likelihoodArray = data(:, lkhdIdx);
    % replace entries in x, y arrays with likelihood < THRESHOLD to NaN
    xArray(likelihoodArray < threshold) = NaN;
    yArray(likelihoodArray < threshold) = NaN;
    % interpolate the entries with NaN
    xArray = repnan(xArray,'linear');
    yArray = repnan(yArray,'linear');
end



%% Plotting related
% plot the difference between a set of original & modified XY, where one
% graph is the original, the next modified, and the last an overlay of
% modified and what changed.
function plot_difference_between_two_arrays(originalX, originalY, ...
    modifiedX, modifiedY, plotName)
    % figure; plot(originalX, originalY);
    % title(append(plotName, " before adjustment"),'Interpreter','none')
    % 
    % figure; 
    % plot(modifiedX, modifiedY, 'b');
    % title(append(plotName, " after adjustment without overlay"), ...
    %     'Interpreter','none')
    % 
    % [interpDiffX, interpDiffY] = change_in_original_array(originalX, ...
    %     modifiedX, originalY);
    % 
    % figure; 
    % plot(modifiedX,   modifiedY,   'b', ...
    %      interpDiffX, interpDiffY, 'r');
    % title(append(plotName, " after adjustment with overlay"), ...
    %     'Interpreter','none')
    NNN = length(originalX);

    figure; plot(originalX(1:NNN), originalY(1:NNN));
    title(append(plotName, " before adjustment"),'Interpreter','none')
    
    figure; 
    plot(modifiedX(1:NNN), modifiedY(1:NNN), 'b');
    title(append(plotName, " after adjustment without overlay"), ...
        'Interpreter','none')

    [interpDiffX, interpDiffY] = change_in_original_array(originalX(1:NNN), ...
        modifiedX(1:NNN), originalY(1:NNN));

    figure; 
    plot(modifiedX(1:NNN),   modifiedY(1:NNN),   'b', ...
         interpDiffX, interpDiffY, 'r');
    title(append(plotName, " after adjustment with overlay"), ...
        'Interpreter','none')
end

% returns the difference in line segments between two x arrays
% and return the corresponding y arrays accordingly as well
function [retX, retY] = change_in_original_array(originalX, modifiedX, ...
    originalY)
    % check for length of two X arrays and Y array
    if (length(originalX) ~= length(modifiedX)) | ...
       (length(originalX) ~= length(originalY))
        disp("Arrays passed to change_in_original_array have " + ...
             "different length!");
        return 
    end
    % otherwise get the difference
    difference = (originalX ~= modifiedX);

    % store_result starts from twice the length of original, which is the
    % longest it can be; we then control length using idx and cut unneeded
    % space at the end
    storeResultX = zeros([1, length(originalX) * 2]); idx = 1;
    storeResultY = zeros([1, length(originalX) * 2]);

    % for every entry in the difference, populate returned accordingly
    for i = 1:length(difference)
        % if value is different at index i:
        if difference(i)
            % populate store_result with the value in original at i
            storeResultX(idx) = originalX(i); storeResultY(idx) = originalY(i); idx = idx + 1;

        % otherwise, if value is not different at index i:
        else
            % if we aren't at the beginning of the array:
            if i ~= 1
                % check what's at i-1 - if it is different, include this
                % to result then further append NaN
                if difference(i - 1)
                    storeResultX(idx) = originalX(i); storeResultY(idx) = originalY(i); idx = idx + 1;
                    storeResultX(idx) = NaN; storeResultY(idx) = NaN; idx = idx + 1;
                % otherwise if i-1 was not different, check i+1
                % if value at i+1 is different, populate store_result
                % with the value in original at i
                elseif i ~= length(difference) && difference(i + 1)
                    storeResultX(idx) = originalX(i); storeResultY(idx) = originalY(i); idx = idx + 1;
                % otherwise, do nothing
                end
            % also deal with situation where we are at beginning of array
            elseif i == 1
                % we check after: if i+1 is different, store result
                if i ~= length(difference) && difference(i + 1)
                    storeResultX(idx) = originalX(i); storeResultY(idx) = originalY(i); idx = idx + 1;
                % otherwise, do nothing
                end
            ...
            end
        ...
        end
    ...
    end

    % finally, we set returned X and Y based on store_result X and Y
    retX = storeResultX(1:idx - 1);
    retY = storeResultY(1:idx - 1);
end

% Render position by speed
function render_position_by_speed(X, Y, nBin)
    speedPerFrame = sqrt(diff(X).^2 + diff(Y).^2);
    maxSpeed = max(speedPerFrame); minSpeed = min(speedPerFrame);

    % get bin minimum values based on max and min speeds
    binWidth = (maxSpeed - minSpeed) / nBin;
    binMins = zeros([1, nBin]) + minSpeed;
    for i = 1:nBin
        binMins(i) = binMins(i) + (i - 1) * binWidth;
    end
    
    % get associated colors to those bins
    binColors = zeros([1, nBin]);
    for i = 1:nBin
        binColors(i) = (i - 1) * 1 / (nBin - 1);
    end

    % open figure first
    figure;
    hold on
    
    % isolate the different bin range values and their rendering
    % using change_in_original_array
    for i = 1:nBin
        binMin = binMins(i); binMax = binMin + binWidth; binColor = binColors(i);
        speedsInRange = (speedPerFrame > binMin & speedPerFrame <= binMax);
        % shift every position within speedsInRange by one to match X/Y
        correspondingPositions = false([1, length(speedsInRange) + 1]);
        correspondingPositions(2:end) = speedsInRange;
        
        filteredX = X; filteredX(correspondingPositions) = NaN;
        % get the difference to be plotted
        [plottedX, plottedY] = change_in_original_array(X, filteredX, Y);
        % plot it with the given color
        
        plot(plottedX, plottedY, Color = [binColor 0 0]);

    end
    
    hold off

end

% Render positions where every other line segment is a different color;
% either c1 or c2
function render_every_other_segment_with_different_color(X, Y, c1, c2)
    if length(X) ~= length(Y)
        disp("Passed arrays X and Y have different lenghts!")
        return
    end
    % two segements - contains NaN every three positions starting at 3 for
    % c1 and 1 for c2
    c1SegmentX = nan([1, length(X) * 1.5]);
    c1SegmentY = nan([1, length(X) * 1.5]);
    c2SegmentX = nan([1, length(X) * 1.5]);
    c2SegmentY = nan([1, length(X) * 1.5]);
    % for every pair i, i+1 in X where i increments by 2 each step
    for i = 1:2:length(X) - 1
        intI = uint32(i);
        xI = X(intI); xIp1 = X(intI+1); yI = Y(intI); yIp1 = Y(intI+1);
        % correspond to i + (i integer division 2) and that+1 in c1SegmentX
        c1SegmentX(intI + idivide(intI,2))     = xI; 
        c1SegmentX(intI + idivide(intI,2) + 1) = xIp1;
        c1SegmentY(intI + idivide(intI,2))     = yI; 
        c1SegmentY(intI + idivide(intI,2) + 1) = yIp1;

        if intI + 2 <= length(X)
            % also get pair i+1, i+2
            xIp2 = X(intI+2); yIp2 = Y(intI+2);
            % correspond to i+1 + (i+1 integer division 2) and that+1 in
            % c2SegmentX
            c2SegmentX(intI+1 + idivide(intI+1,2) + 1) = xIp2;
            c2SegmentY(intI+1 + idivide(intI+1,2) + 1) = yIp2;
        end

        c2SegmentX(intI+1 + idivide(intI+1,2))     = xIp1;
        c2SegmentY(intI+1 + idivide(intI+1,2))     = yIp1;    
    end

    figure;
    plot(c1SegmentX, c1SegmentY, Color = c1);
    hold on
    plot(c2SegmentX, c2SegmentY, Color = c2);
    hold off
end


%% Sanity checks
% does an easy sanity check for filter_by_speed_upperbound
function sanity_check_filter_speed_by_upperbound()
    X = [1,2,3,14,15,6,7,8,9,210,302,12,13];
    Y = [1,2,3,14,15,6,7,8,9,210,302,12,13];

    upperBound = 100;

    expX = [1,2,3,14,15,6,7,8,9,10,11,12,13];
    expY = [1,2,3,14,15,6,7,8,9,10,11,12,13];

    [actX, actY] = filter_by_speed_upperbound(X, Y, upperBound);
    assert(prod(actX == expX) * prod(actY == expY));

    upperBound = 15;

    expX = [1,2,3,4,5,6,7,8,9,10,11,12,13];
    expY = [1,2,3,4,5,6,7,8,9,10,11,12,13];

    [actX, actY] = filter_by_speed_upperbound(X, Y, upperBound);
    assert(prod(actX == expX) * prod(actY == expY));
end

% does an easy sanity check for change_in_original_array
function sanity_check_change_in_original_array(plotFigures)
    origArrX     = [  0,   1,  2,  3,   4,   5,   6,   7,  8];
    newArrX      = [  0,   1,  2, 93,   4,   5,  96,   7,  8];
    origArrY     = [  8,   7,  6,  5,   4,   3,   2,   1,  0];
    newArrY      = [  8,   7,  6, 95,   4,   3,  92,   1,  0];

    expDiffX = [2, 3, 4, NaN, 5, 6, 7, NaN];
    expDiffY = [6, 5, 4, NaN, 3, 2, 1, NaN];
    
    [diffArrX, diffArrY] = change_in_original_array(origArrX, newArrX, origArrY);
    assert( prod( expDiffX(~isnan(expDiffX)) == diffArrX(~isnan(diffArrX)) ) * ...
            prod( isnan(expDiffX) == isnan(diffArrX)));
    assert( prod( expDiffY(~isnan(expDiffY)) == diffArrY(~isnan(diffArrY)) ) * ...
            prod( isnan(expDiffY) == isnan(diffArrY)));
    
    if plotFigures
        figure;
        plot(origArrX, origArrY, 'b');
        title("OnlyBlue",'Interpreter','none')
        
        figure;
        plot(origArrX, origArrY, 'b', ...
             diffArrX, diffArrY, 'r');
        title("Both",'Interpreter','none')
    end
end


function reduction_in_distance(X, Y, filteredX, filteredY)
    sumDistanceTraveledBeforeFiltering = sum(sqrt(diff(X).^2 + ...
                                                  diff(Y).^2));
    sumDistanceTraveledAfterFiltering  = sum(sqrt(diff(filteredX).^2 + ...
                                                  diff(filteredY).^2));
    disp ("We reduce the amount of total movement by this much % by " + ...
    "excluding outliers: " + ...
    (sumDistanceTraveledBeforeFiltering - ...
     sumDistanceTraveledAfterFiltering) / ...
    sumDistanceTraveledBeforeFiltering * 100);
end

