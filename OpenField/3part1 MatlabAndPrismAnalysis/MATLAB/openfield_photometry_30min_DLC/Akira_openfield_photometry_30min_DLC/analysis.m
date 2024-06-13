%% Defining the analysis function
% We define the function which analyzes a csv given with path and stores 
% results - both in individual folders and in one file "analysis_data".
% Currently separated so that we can run a test on them - but ultimately
% I aim to merge it back into one script file to reduce confusion.

% xmax, xmin, ymax, ymin - Used to specify the max & min positions of the 
%  cage sides in coordinates. If given as NaN, will use the max & min of 
%  the snout X&Y arrays. Put here in order to allow correct computation of
%  cases where the mouse didn't cover the entire cage.
function analysis(csvPath,saveDir,graphTitle,resultData,skipUserPrompt, ...
    displayFigure, xmax, xmin, ymax, ymin)

    %% Define useful constants
    SKIP_SAVING = true;

    %% Get file and plot mouse position over time using TailBase position
    % Video used to be analyzed from 0:30 ~ 30:30
    % However, for behavior analysis, I would use 0:00 ~ 30:00 as some
    % videos are not long enough.
    % If file is shorter, user is prompted whether to truncate and proceed 
    %  Do note it was shorter!

    disp("!!!!CURRENTLY WE COMPUTE VALUES BETWEEN 0:00 AND 30:00 FOR BEHAVIOR!!!!");
    
    % Opening file
    data = readmatrix(csvPath);
    
    % Obtain data using function defined at end of file
    TAILBASE_X_IDX = 17; TAILBASE_Y_IDX = 18; TAILBASE_LIKELIHOOD_IDX = 19;
    BELLY_X_IDX = 20;    BELLY_Y_IDX = 21;    BELLY_LIKELIHOOD_IDX = 22;
    SNOUT_X_IDX = 2;     SNOUT_Y_IDX = 3;     SNOUT_LIKELIHOOD_IDX = 4;
    
    % Position of Tail Base, Belly, Snout
    [TailBaseX, TailBaseY] = computeinterpolatedxycoordinates( ...
        TAILBASE_X_IDX, TAILBASE_Y_IDX, TAILBASE_LIKELIHOOD_IDX, data);
    
    [BellyX, BellyY]       = computeinterpolatedxycoordinates( ...
        BELLY_X_IDX, BELLY_Y_IDX, BELLY_LIKELIHOOD_IDX, data);
    
    [SnoutX, SnoutY]       = computeinterpolatedxycoordinates( ...
        SNOUT_X_IDX, SNOUT_Y_IDX, SNOUT_LIKELIHOOD_IDX, data);
    
    % Rewrite time based on FRAMES_PER_SECOND frames per second
    FRAMES_PER_SECOND = 40;
    
    Time = data(:,1)/FRAMES_PER_SECOND;
    
    TIME_START = 1; % from 0 seconds
    TIME_END = 1800 * FRAMES_PER_SECOND; % to 30:00
    
    % check video has a certain length, flag if falling short
    if length(Time) < TIME_END
        disp("This file has " + length(Time) + ...
             " frames which is less than what specified by TIME_END: " + ...
             TIME_END + ".");
        % prompt user whether to proceed
        % if skipUserPrompt is true, simply skip
        if (skipUserPrompt)
            disp("skipUserPrompt is true - proceeding with shorter video!");
            TIME_END = length(Time);
        elseif (propmt_question("Would you like to proceed? y/n: "))
            % if they agree, set TIME_END to correct bound
            TIME_END = length(Time);
        else % otherwise throw error
            ME = MException('OpenfieldPhotometry30min:videoIsTooShort', ...
                'Video %s had less frames than expected - aborting.',file);
            throw(ME);
        end
    end
    
    % Clip arrays to appropriate range
    Time =           Time(TIME_START:TIME_END);
    TailBaseX = TailBaseX(TIME_START:TIME_END);
    TailBaseY = TailBaseY(TIME_START:TIME_END);
    BellyX =       BellyX(TIME_START:TIME_END);
    BellyY =       BellyY(TIME_START:TIME_END);
    SnoutX =       SnoutX(TIME_START:TIME_END);
    SnoutY =       SnoutY(TIME_START:TIME_END);

    % Calibrate [xmax,xmin,ymax,ymin] as appropriate
    if ~isnan(xmax) && ~isnan(xmin) && ~isnan(ymax) && ~isnan(ymin)
        disp("All of xmax,xmin,ymax,ymin were provided successfully.")
        disp("These values will be used to calculate the boundaries of the cage!")
    else % otherwise, we use the positions of the snout to find the boundaries of the cage
        xmax = max(SnoutX); xmin = min(SnoutX); 
        ymax = max(SnoutY); ymin = min(SnoutY);
    end
    
    % Plot & save figures using function defined at end of script
    tailBasePlotName = append(graphTitle, ' Tail Base Coordinates');
    bellyPlotName    = append(graphTitle, ' Belly Coordinates');
    snoutPlotName    = append(graphTitle, ' Snout Coordinates');

    if ~SKIP_SAVING
        % TailBase positions 
        render_and_save_plot(TailBaseX, TailBaseY, tailBasePlotName, ... %graphTitle, ...
            fullfile(saveDir, 'Position Graphs'), displayFigure); 
        % Belly positions
        render_and_save_plot(BellyX, BellyY, bellyPlotName, ... %graphTitle, ...
            fullfile(saveDir, 'Position Graphs'), displayFigure); 
        % Snout positions
        render_and_save_plot(SnoutX, SnoutY, snoutPlotName, ... %graphTitle, ...
            fullfile(saveDir, 'Position Graphs'), displayFigure);
    end
     
    %% Calculate total distance travelled
    % Using tailbase position
    
    % compute difference between consecutive entries of TailBase arrays
    xdiff = diff(TailBaseX);
    ydiff = diff(TailBaseY);
    % compute euclidean distance from xdiff and ydiff values
    totaldiff = sqrt(xdiff.^2 + ydiff.^2);
    % sum all differences to get the total distance covered by mouse
    totaldistance = sum(totaldiff,'omitnan');
    %totaldistancepartial = sum(totaldiff(end-11400:end),'omitnan');
    disp 'total distance in pixels is'
    disp(totaldistance);
    % save the result
    if ~SKIP_SAVING
        save_variable('totaldistance','Total distance pixels',saveDir,graphTitle)    
    end
    
    
    %% Calculate ranges, convert ranges to cm
    % Calculate total distance travelled in cm

    % these names are my guess...
    CAGE_X_SIDE_LENGTH_CM = 38;
    CAGE_Y_SIDE_LENGTH_CM = 38;
    CAGE_DIAGONAL_LENGTH_CM = 53.7401153702;
    
    % get the range of x / y pixel values belonging to the cage 
    xrange = xmax - xmin; yrange = ymax - ymin;

    % get the length of the diagonal of the rectangle the mice moved in
    totalrange = sqrt(xrange.^2 + yrange.^2);
    % calculate the size of cm/pixel in both directions
    % I DON'T LIKE THAT THIS IS VIDEO DEPENDENT - BETTER WAYS?
    x_cm = CAGE_X_SIDE_LENGTH_CM/xrange;
    y_cm = CAGE_Y_SIDE_LENGTH_CM/yrange;
    total_cm = CAGE_DIAGONAL_LENGTH_CM/totalrange;
    xdiff_cm = xdiff*x_cm;
    ydiff_cm = ydiff*y_cm;
    % DOUBLE CHECK
    % Is below correct? Don't we need to take the square of cm diffs? 
    totaldiff_cm = abs(totaldiff*total_cm);
    
    % totaldistance_cm_from_square = sum(sqrt(xdiff_cm.^2 + ydiff_cm.^2))

    totaldistance_cm = sum(totaldiff_cm,"omitnan");

    disp 'total distance in cm is'; disp(totaldistance_cm);
    
    if ~SKIP_SAVING
        save_variable('totaldiff_cm','Total diff cm',saveDir,graphTitle)
        save_variable('totaldistance_cm','Total distance cm',saveDir,graphTitle)
        % Not quite sure what Total distance cm partial is - same as
        % totaldistance_cm?
        save_variable('totaldistance_cm','Total distance cm partial',saveDir,graphTitle)
    end
    
    %% Calculate center time
    % convert pixels coordinates to cm coordinates
    xmax_cm = xmax*x_cm; xmin_cm = xmin*x_cm;
    ymax_cm = ymax*y_cm; ymin_cm = ymin*y_cm;
    BellyX_cm = BellyX*x_cm; BellyY_cm = BellyY*y_cm;
    
    % determine the max&min in cm coordinates that qualify as "center"
    DISTANCE_CM_FROM_EDGE_X = 8; DISTANCE_CM_FROM_EDGE_Y = 8;
    xcentermin = (xmin_cm + DISTANCE_CM_FROM_EDGE_X); 
    xcentermax = (xmax_cm - DISTANCE_CM_FROM_EDGE_X);

    ycentermin = (ymin_cm + DISTANCE_CM_FROM_EDGE_Y); 
    ycentermax = (ymax_cm - DISTANCE_CM_FROM_EDGE_Y);
    
    % count the number of frames where Belly is within "center"
    centerTime = sum(xcentermin <= BellyX_cm & BellyX_cm <= xcentermax & ...
                     ycentermin <= BellyY_cm & BellyY_cm <= ycentermax, ...
                     "omitnan");
    % convert value into percentage
    centerTime = centerTime / length(Time) * 100;
    
    if ~SKIP_SAVING
        save_variable('centerTime','Center time',saveDir,graphTitle)
    end
    
    disp 'Center time is'; disp(centerTime);

    %% Calculate time percentage spent in center per 5 minutes
    FRAMES_PER_FIVE_MINUTES = 5 * 60 * FRAMES_PER_SECOND; % 12000 when 40Hz
    
    % we compute center time (and later total distance) for the intervals 
    % number between START_INTERVAL_NUMBER and END_INTERVAL_NUMBER
    %  e.g. with START_INTERVAL_NUMBER = 1 and END_INTERVAL_NUMBER = 6 
    %  respectively, we compute for the first to sixth interval
    START_INTERVAL_NUMBER = 1; END_INTERVAL_NUMBER = 6;

    % obtain a series of 0/1 for frames in and out of center
    isAtCenter = xcentermin <= BellyX_cm & BellyX_cm <= xcentermax & ...
                 ycentermin <= BellyY_cm & BellyY_cm <= ycentermax;
    % obtain how many frames this was for each interval
    centerTimeByIntervals= computetotalperinterval( ...
     isAtCenter, FRAMES_PER_FIVE_MINUTES, START_INTERVAL_NUMBER, ...
     END_INTERVAL_NUMBER ...
     );

    % convert into percentage by dividing each interval by their length
    intervalLength = computetotalperinterval( ...
     ones(length(isAtCenter)), FRAMES_PER_FIVE_MINUTES, ...
     START_INTERVAL_NUMBER, END_INTERVAL_NUMBER ...
     );
    centerTimeByIntervals= centerTimeByIntervals./ intervalLength * 100;
    
    if ~SKIP_SAVING
        save_variable('centerTimeByIntervals','Center time by intervals', ...
            saveDir,graphTitle)
    end
    
    disp 'Center time by intervals is'; disp(centerTimeByIntervals);
    
    %% Calculate time spent in each of 4 quadrants
    % Quadrants increment clockwise from bottom left quadrant (Quad1) 
    % get the middle point
    xmid = xmin + xrange/2; ymid = ymin + yrange/2;
    
    % left bottom, left top, right top, right bottom
    timeQuad1 = sum(BellyX <  xmid & BellyY <= ymid, "omitnan") / FRAMES_PER_SECOND;
    timeQuad2 = sum(BellyX <= xmid & BellyY >  ymid, "omitnan") / FRAMES_PER_SECOND;
    timeQuad3 = sum(BellyX >  xmid & BellyY >= ymid, "omitnan") / FRAMES_PER_SECOND;
    timeQuad4 = sum(BellyX >= xmid & BellyY <  ymid, "omitnan") / FRAMES_PER_SECOND;
    
    timeByQuad = [timeQuad1, timeQuad2, timeQuad3, timeQuad4];
    % also get percent ratio per quadrant
    timeByQuadFrac = timeByQuad / sum(timeByQuad, "omitnan") * 100;

    if sum(timeByQuad, "omitnan") - length(BellyX)/FRAMES_PER_SECOND > 0.1
        disp("WARNING: Times don't add to 60 minutes; Values not saved");
    elseif ~SKIP_SAVING
        save_variable('timeByQuad','Time by quadrants',saveDir,graphTitle)
        save_variable('timeByQuadFrac','Fraction of time by quadrants', ...
            saveDir,graphTitle)
    end
    
    %% Calculate total distance per 5 minutes
    
    % obtain the pixel distance for each interval
    pixelDistance = computetotalperinterval( ...
     totaldiff, FRAMES_PER_FIVE_MINUTES, START_INTERVAL_NUMBER, ...
     END_INTERVAL_NUMBER ...
     );
        
    % convert it into cm distance
    distanceByIntervals = abs(pixelDistance * total_cm);
 
    if ~SKIP_SAVING
        save_variable('distanceByIntervals', ...
            'Total distance per interval',saveDir,graphTitle)
    end
    
    %% Calculate mouse movement and speed
    % Removes impossible speeds by filling outliers
    % Based on 40 Hz frame rate
    
    %xspeed = abs(xdiffinches * FRAMES_PER_SECOND);
    %yspeed = abs(ydiffinches * FRAMES_PER_SECOND );
    totalspeed = abs(totaldiff_cm * FRAMES_PER_SECOND);
    
    averagespeed = mean(totalspeed,'omitnan');
    disp('average speed is');
    disp(averagespeed);

    if ~SKIP_SAVING
        render_and_save_plot(Time(2:end), totalspeed, graphTitle, ...
            fullfile(saveDir, 'Speed plots'), displayFigure);
        
        save_variable('totalspeed','Speed over time',saveDir,graphTitle)
        save_variable('averagespeed','Average speed',saveDir,graphTitle)
    end
    
    %% Storing result data
    % Finally, store all result data into the ResultData object passed in

    % Identify mouse type (e.g. YAC128, WT...) based on csvPath
    % Assume that files were direct children of a folder named with the 
    % mouse type (e.g. something like /WildType/this_csv.csv)
    splitPath = split(csvPath, filesep);
    mouseType = splitPath(end-1);

    newData = struct("mouseType", mouseType,...
                     "totalDistanceCm", totaldistance_cm,...
                     "centerTime", centerTime,...
                     "centerTimeByIntervals", centerTimeByIntervals,...
                     "timeFractionByQuadrant", timeByQuadFrac,...
                     "distanceByIntervals", distanceByIntervals...
                     );
    
    resultData.adddata(convertCharsToStrings(graphTitle), newData);

    %% Definition of helper functions
    
    % Produces arrays of x & y coordinates stored in xIdx and yIdx
    % If corresponding likelihood (stored in lkhdIdx) of a coordinate is 
    % less than THRESHOLD, interpolate it instead using the repnan function
    function [xArray, yArray] = computeinterpolatedxycoordinates( ...
        xIdx, yIdx, lkhdIdx, data ...
        )
        % threshold of likelihood to interpolate given point
        THRESHOLD = 0.8;
        xArray = data(:,xIdx);
        yArray = data(:,yIdx);
        likelihoodArray = data(:, lkhdIdx);
        % replace entries in x, y arrays with likelihood < THRESHOLD to NaN
        xArray(likelihoodArray < THRESHOLD) = NaN;
        yArray(likelihoodArray < THRESHOLD) = NaN;
        % interpolate the entries with NaN
        xArray = repnan(xArray,'linear');
        yArray = repnan(yArray,'linear');
    end

    % Compute total of given series per 5 minutes interval
    % interval_length : int, number of frames per interval
    % init_interval, last_interval : int, we compute total 
    %  of series for the intervals number between them
    %  e.g. with init_interval = 1 and last_interval = 6 
    %  respectively, we compute for the first to sixth interval
    function [seriesPerInterval] = computetotalperinterval( ...
        series, interval_length, init_interval, last_interval ...
        )
        seriesPerInterval = zeros(1, last_interval - init_interval + 1);
    
        for i=init_interval:last_interval
            % set the beginning of the interval based on i
            % for i = 1 only, the beginning is 1 instead of 0
            intervalStart = interval_length * (i - 1);
            if (intervalStart <= 0)
                intervalStart = 1;
            end
            % set the end of interval based on i
            intervalEnd = interval_length * i - 1;
            % if intervalEnd exceed the actual video length, truncate
            if length(series) < intervalEnd
                disp("Not enough frames for the " + i + "th interval of the " + ...
                    "video: " + length(series) + " instead of expected " + ... 
                    intervalEnd + " - truncating.");
                intervalEnd = length(series);
            end
    
            % obtain the total of the series for interval i
            intervalTotal = sum(series(intervalStart:intervalEnd), 'omitnan');
            % add result into the array of series by interval
            seriesPerInterval(1,i) = intervalTotal;
        end
    end
    
    % Prints the specified x & y coordinate arrays into a plot and saves
    % it to saveDir, while storing them into the specific folders
    function render_and_save_plot(xArray, yArray, graphTitle, saveDir, ...
            displayFigure)
        
        % create figure and plot the coordinates
        if displayFigure
            figure;
        else % invisible if displayFigure is false
            figure('visible', 'off');
        end    
        plot(xArray, yArray);
        % set the title of the graph
        title(graphTitle,'Interpreter','none')
        % if saveDir does not exist a directory, create it
        if ~exist(saveDir, 'dir')
           disp(append("Folder: ", saveDir, " did not " + ...
                "exist - created it!"))
           mkdir(saveDir)
        end
        % save the graph at specified positions
        saveas(gcf,fullfile(saveDir, append(graphTitle,'.png')));
    end

    % Saves the given variable into the specified folder under given 
    %  saveDir with known graphTitle and in -ascii mode.
    % If saveFolder doesn't exist under saveDir, create it.
    function save_variable(variable,saveFolder,saveDir,graphTitle)
        fullPathToFolder = fullfile(saveDir, saveFolder);
        % if saveFolder doesn't exist under saveDir, create it
        if ~exist(fullPathToFolder, 'dir')
            disp(append("Folder: ", fullPathToFolder, " did not " + ...
                "exist - created it!"))
            mkdir(fullPathToFolder)
        end
        % then save the content
        save( fullfile(fullPathToFolder, graphTitle), variable, '-ascii');
    end
    
    % Prompts a question, and recognizes any entries indicating a yes and no
    % Return a boolean corresponding to true = "yes", false = "no".
    % If user inputs anything else, ask again
    function answeredYes = propmt_question(message)
        % list of inputs recognizes as a "yes" and "no"
        inputRecognizedAsYes = ['y','Y','yes','YES'];
        inputRecognizedAsNo  = ['n','N', 'no', 'NO'];
        
        while (true)
            % prompt the user
            txt = input(message,"s");
            
            % if input is recognized as yes or no, return those
            if (ismember(txt, inputRecognizedAsYes)) 
                answeredYes = true; return
            elseif (ismember(txt, inputRecognizedAsNo))
                answeredYes = false; return
            % otherwise, prompt again
            else
                disp("Answer wasn't successfully parsed - try again!");
            end
        end
    end


end