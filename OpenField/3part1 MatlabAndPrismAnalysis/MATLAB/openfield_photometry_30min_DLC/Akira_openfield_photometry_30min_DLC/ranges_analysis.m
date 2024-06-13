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

    % Opening file
    data = readmatrix(csvPath);
    
    % Obtain data using function defined at end of file
    TAILBASE_X_IDX = 17; TAILBASE_Y_IDX = 18; TAILBASE_LIKELIHOOD_IDX = 19;
    BELLY_X_IDX = 20;    BELLY_Y_IDX = 21;    BELLY_LIKELIHOOD_IDX = 22;
    SNOUT_X_IDX = 2;     SNOUT_Y_IDX = 3;     SNOUT_LIKELIHOOD_IDX = 4;
    
    % Position of Tail Base, Belly, Snout
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
     
    % get the range of x / y pixel values belonging to the cage 
    xrange = xmax - xmin; yrange = ymax - ymin;
    % get the length of the diagonal of the rectangle the mice moved in
    totalrange = sqrt(xrange.^2 + yrange.^2);

    disp("xmax: ", xmax, "xmin: ", xmin)
    disp("ymax: ", ymax, "ymin: ", ymin)
    disp("xrange: ", xrange)
    disp("yrange: ", yrange)
    disp("totalrange: ", totalrange)

    %% Storing result data
    % Finally, store all result data into the ResultData object passed in

    % Identify mouse type (e.g. YAC128, WT...) based on csvPath
    % Assume that files were direct children of a folder named with the 
    % mouse type (e.g. something like /WildType/this_csv.csv)
    splitPath = split(csvPath, filesep);
    mouseType = splitPath(end-1);

    newData = struct("mouseType", mouseType,...
                     "xmax", xmax, ... 
                     "xmin", xmin, ...
                     "ymax", ymax, ...
                     "ymin", ymin, ...
                     "xrange", xrange, ...
                     "yrange", yrange, ...
                     "totalrange", totalrange...
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