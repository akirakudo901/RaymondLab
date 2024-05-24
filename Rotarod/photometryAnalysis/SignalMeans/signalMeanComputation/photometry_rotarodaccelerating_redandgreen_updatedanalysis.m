% Summary: This script imports the TDT photometry files for red, green, and
% isosbestic channel data. It takes out the first 30s of the trial. The
% green isosbestic signal is used to calculate dF/F for the green channel.
% The red channel uses a 2nd polynomial fit to create the baseline and then
% this is used to calculate dF/F (new method). The before & after phases of the trial
% are 2.5 minutes long (shorter than in previous analysis). The 2.5 minutes
% before mouse is put on rotarod is used as the baseline for Z score
% calculation. This script also calculates the average Z score for both red
% and green channels during the different phases of the trial.

%Script created by Ellen Koch and uses functions/code from TDT and Jaideep Bains
%and colleagues.
% 
% recalculate : If true, recalculate and overwrite existing entries within 
% resultData with the same name as the current data.

% We also deal with a few exceptions:
% 1) Data that doesn't last long enough to calculate the "After" values for
%    signals once the mouse falls off the rotarod. 
%    -> Displays a message, and uses available data for calculation. Also,
%       inscribe the fact into the CSV, under "After_TooShort".
% 2) How many data.epocs.Note.onset & data.epocs.PtAB.onset data entries a 
%    given info file had
% 3) If an exception is raised, what it was
% 4) Other info added manually
function data = photometry_rotarodaccelerating_redandgreen_updatedanalysis( ...
    dataDir, resultData, recalculate, ...
    SAVE_JPG, SAVE_EMF, FIGURE_SAVE_DIR, DISPLAY_FIGURES, DISPLAY_COMMENTS)
    dataDir = convertStringsToChars(dataDir);

    % add path to helpers
    addpath(fullfile(".", "helpers"));
    
    %% Useful constants
    % files with these names hold data with updated time stamps so we load 
    % them instead of other data
    FILES_TO_LOAD_IF_EXISTING_IN_FOLDER = [
        "Updated Notes and PtAB.mat", "Updated Notes.mat", ... 
        "Updated PrtB and Notes.mat", "Updated PtAB.mat"];


    %% Execution
    % get and process mouse name
    [mouseName,mouseNameAll,day,trialInDay,totalTrial] = ...
        process_mouse_name(dataDir);
    trialInDay = trialInDay(1); % we might need just the trial number

    % if an entry with the same data is in resultData and parameter 
    % 'recalculate' is false, skip this calculation
    if ~recalculate && resultData.iskey(mouseNameAll)
        if DISPLAY_COMMENTS
            disp("Entry with same name found in resultData, skipping!");
        end
        return
    end

    %% Import data from TDT files or pre-stored mat file
    % When we have files named "Updated Notes and PtAB.mat", 
    % "Updated Notes.mat", "Updated PrtB and Notes.mat", or 
    % "Updated PtAB.mat" under dataDir, load it. This holds a
    % structure named "data" with (presumably) correct timestamps.
    if DISPLAY_COMMENTS
        disp("-------------------------------------------")
    end

    % check if files of the above form are there
    checkedDirs = dir(dataDir)';
    % check we only have one file to load with given names; if more, error
    foundOneFileToLoad = false;
    for file = FILES_TO_LOAD_IF_EXISTING_IN_FOLDER
        if ismember(file, {checkedDirs.name})
            if foundOneFileToLoad
                ME = MException(Constants.ErrorIdentifierPrefix, ':', ...
                    'moreThanTwoFilesToLoad', ...
                    ['Following folder has more than two files to load updated' ...
                    'timestamps from: %s.'], dataDir);
                throw(ME);
            else 
                foundOneFileToLoad = true;
                if DISPLAY_COMMENTS
                    fprintf("Filename: %s.\n", dataDir);
                    fprintf("Found an updated timestamp version from file: %s - loading!", ...
                        file);
                    disp(" ");
                end
            end
            % load data into an object named 'data'
            data = load(fullfile(dataDir, file)); data = data.data;
            % if data.epocs.PtAB is directly a vector rather than a struct 
            % on its own (as it seems to happen for example when loading a
            % "Updated PtAB.mat" file), we rearrange so that 
            % data.epocs.PtAB.onset is a vector.
            if ~isstruct(data.epocs.PtAB)
                % this is a very basic struct emulating the "onset"
                % entry of usual PtAB structs but nothing else
                basicPtABstruct = struct("onset", data.epocs.PtAB);
                data.epocs.PtAB = basicPtABstruct;
            end
        end
    end
    % if unable to find any files to load from, load data using TDTbin2mat
    if ~foundOneFileToLoad
        data = TDTbin2mat(dataDir);
    end

    % Define a function that adds an empty entry with the given stats
    function addemptydata(resultData, day, trialInDay, mouseName, ...
            totalTrial, message, mouseNameAll)
        newData = struct( ...
                "Day", day, ...
                "Trial", trialInDay, ...
                "Means_Green", '', ...
                "Meanshift_Green", '', ...
                "Means_Red", '', ...
                "Meanshift_Red", '',  ...
                "MouseName", mouseName, ...
                "TotalTrial", totalTrial, ...
                "After_TooShort", '', ...
                "Note_Onset_Size1", '', ...
                "Note_Onset_Size2", '', ...
                "PtAB_Onset_Size1", '', ...
                "PtAB_Onset_Size2", '', ...
                "Exception", message, ...
                "Info", ''...
                );

        resultData.adddata(append("Failed_", mouseNameAll), ...
                           newData);
    end
    %% Check weird data cases
    % 1 - Error: 'Unrecognized field name "Note".' (OR PtAB)
    % Occurs when there is no onset/offset entry in the Notes.txt.
    % Often with trials that were restarted.
    try
        data.epocs.Note.onset(1,1);
        data.epocs.PtAB.onset(1,1);
    catch ME
        if strcmp(ME.message, "Unrecognized field name ""Note"".") || ...
           strcmp(ME.message, "Unrecognized field name ""PtAB"".")
            if strcmp(ME.message, "Unrecognized field name ""Note"".")
                lacking_entry = "Note";
            elseif strcmp(ME.message, "Unrecognized field name ""PtAB"".")
                lacking_entry = "PtAB";
            end
            if DISPLAY_COMMENTS
                disp("This file apparently has no '" + lacking_entry +  ...
                    "' entry - likely a failed attempt. Skipping.");
            end
            
            % save an empty result into resultData
            addemptydata(resultData, day, trialInDay, mouseName, ...
                totalTrial, ME.message, mouseNameAll)
            return
        end
    end
    % 2 - When we have less than two entries for either of:
    % - data.epocs.Note.onset 
    % - data.epocs.PtAB.onset
    % Trials that failed to have good recordings?
    try
        data.epocs.Note.onset(2,1);
    catch ME
        if contains(ME.message, "Index must not exceed")
            if DISPLAY_COMMENTS
                disp("This file apparently has not enough 'Note.onset' " +  ...
                    "entry - likely a failed attempt. Skipping.");
            end
            % save an empty result into resultData
            addemptydata(resultData, day, trialInDay, mouseName, ...
                totalTrial, "Not enough 'Note.onset' entries.", mouseNameAll);
            return
        end
    end

    try
        data.epocs.PtAB.onset(2,1);
    catch ME
        if contains(ME.message, "Index must not exceed")
            if DISPLAY_COMMENTS
                disp("This file apparently has not enough 'PtAB.onset' " +  ...
                    "entry - likely a failed attempt. Skipping.");
            end
            % save an empty result into resultData
            addemptydata(resultData, day, trialInDay, mouseName, ...
                totalTrial, "Not enough 'PtAB.onset' entries.", mouseNameAll);
            return
        end
    end
    
    %% Create time vector from data
    % Fs = sampling rate
    % tssub = take out first 30 seconds of data
    Fs = data.streams.x65G.fs;
    TAKEOUT_N_INITAL_SECONDS = 30;
    TAKEOUT_N_INITIAL_FRAMES = round(TAKEOUT_N_INITAL_SECONDS*Fs);
    
    ts = (0:(length(data.streams.x65G.data) - 1)) / Fs;
    tssub = ts(TAKEOUT_N_INITIAL_FRAMES:end);

    %% Obtain Rotarod key moments saved as seconds
    % (Akira): Noise can cause rotarod & notepoint timestamps to be more 
    %          than 2 entries. In this case, Ellen advised me to pick the
    %          last two entries for each (which should correctly be the
    %          signals we want). Proceeding accordingly to this!
    Pickup_Seconds = data.epocs.Note.onset(end-1,1); % pickup should be the second last of Note.onset
    Start_Seconds  = data.epocs.PtAB.onset(end-1,1); % start the second last of rotarod
    Stop_Seconds   = data.epocs.PtAB.onset(end,1);   % stop the last of rotarod
    Down_Seconds   = data.epocs.Note.onset(end,1);   % down the last of notepoints
    
    %% Extract signal and control data and create plot
    % Apply lowpass filter 
    green = data.streams.x65G.data(TAKEOUT_N_INITIAL_FRAMES:end);
    red = data.streams.x60R.data(TAKEOUT_N_INITIAL_FRAMES:end);

    ctr_green = data.streams.x05G.data(TAKEOUT_N_INITIAL_FRAMES:end);
    
    green=double(green); red=double(red); ctr_green=double(ctr_green);
    
    ctr_green_filt=lowpassphotometry(ctr_green,Fs,2);
    
    if DISPLAY_FIGURES
        figure;
    else % invisible if displayFigure is false
        figure('visible', 'off');
    end
    min_len = min(length(tssub), length(ctr_green_filt));
    plot(tssub(1:min_len), ctr_green_filt(1:min_len))
        
    hold on
    min_len = min(length(tssub), length(green));
    plot(tssub(1:min_len), green(1:min_len))
    min_len = min(length(tssub), length(red));
    plot(tssub(1:min_len), red(1:min_len))
    xlabel('Time(s)','FontSize',14)
    ylabel('mV at detector','FontSize',14)
    daspect([1 1 1])
    hold off
    
    folderName = 'Raw data figure'; graphTitle = append(mouseNameAll, " ", folderName);
    save_figure(gcf, folderName, FIGURE_SAVE_DIR, graphTitle, SAVE_JPG, SAVE_EMF);
    
    if ~DISPLAY_FIGURES 
        close(gcf); 
    end
    
    %% Plot of all 3 signals separately
    if DISPLAY_FIGURES
        figure('units','normalized','outerposition',[0 0 1 1])
    else % invisible if displayFigure is false
        figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
    end

    subplot(3,1,1);
    min_len = min(length(tssub), length(green));
    plot(tssub(1:min_len), green(1:min_len), 'color', [0 0.5 0]);
    xlabel('Time(s)','FontSize',14)
    ylabel('D1-GCAMP','FontSize',14)
    %ylim([mean(green)-20 mean(green)+20])
    title(data.info.blockname(1,:),'Interpreter','none','FontSize',16)
    %vline([Pickup_Seconds Down_Seconds data.epocs.Note.onset(3,1) data.epocs.Note.onset(4,1)] ,{'k', 'k', 'k', 'k'});

    hold on
    subplot(3,1,2);
    min_len = min(length(tssub), length(red));
    plot(tssub(1:min_len), red(1:min_len), 'color', 'r');
    xlabel('Time(s)','FontSize',14)
    ylabel('D2-RCAMP','FontSize',14)
    %ylim([mean(red)-5 mean(red)+10])
    %vline([Pickup_Seconds Down_Seconds data.epocs.Note.onset(3,1) data.epocs.Note.onset(4,1)] ,{'k', 'k', 'k', 'k'});

    hold on
    subplot(3,1,3);
    min_len = min(length(tssub), length(ctr_green_filt));
    plot(tssub(1:min_len), ctr_green_filt(1:min_len), 'color', 'm');
    %ylim([40 80]);
    xlabel('Time(s)','FontSize',14)
    ylabel('Green Isosbestic','FontSize',14)
    %ylim([220 370]);
    %vline([Pickup_Seconds Down_Seconds data.epocs.Note.onset(3,1) data.epocs.Note.onset(4,1)] ,{'k', 'k', 'k', 'k'});
    hold off
    
    folderName = 'Raw data all 4 channels with notes';
    graphTitle = append(mouseNameAll, " ", folderName);
    save_figure(gcf, folderName, FIGURE_SAVE_DIR, graphTitle, SAVE_JPG, SAVE_EMF);

    if ~DISPLAY_FIGURES 
        close(gcf); 
    end
    
    %% Create red control channel
    x = 1:length(red);
    p = polyfit(x, red, 2);
    ctr_red = polyval(p,x);
    
    if DISPLAY_FIGURES
        figure;     
    else % invisible if displayFigure is false         
        figure('visible', 'off'); 
    end

    min_len = min(length(tssub), length(red));
    plot(tssub(1:min_len), red(1:min_len), 'color', 'r');
    hold on
    min_len = min(length(tssub), length(ctr_red));
    plot(tssub(1:min_len), ctr_red(1:min_len), 'color', 'b');
    title('Red channel polyfit');
    hold off
    
    folderName = 'Red figure with control channel line';
    graphTitle = append(mouseNameAll, " ", folderName);
    save_figure(gcf, folderName, FIGURE_SAVE_DIR, graphTitle, SAVE_JPG, SAVE_EMF);
    
    if ~DISPLAY_FIGURES 
        close(gcf); 
    end

    %% Normalize red and green signals to control channels
    normRed = (red - ctr_red)./ctr_red * 100;
    [normGreen] = deltaFF(green,ctr_green_filt);
    
    if DISPLAY_FIGURES
        figure('units','normalized','outerposition',[0 0 1 1]);
    else % invisible if displayFigure is false
        figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
    end
    
    subplot(2,1,1);
    min_len = min(length(tssub), length(normGreen));
    plot(tssub(1:min_len), normGreen(1:min_len), 'color', [0 0.5 0]);
    xlabel('Time(s)','FontSize',14)
    ylabel('D1-GCAMP dF/F','FontSize',14)
    title(data.info.blockname(1,:),'Interpreter','none')
    vline([Pickup_Seconds Down_Seconds Start_Seconds Stop_Seconds], {'k', 'k', 'g', 'g'});
    
    hold on
    subplot(2,1,2);
    min_len = min(length(tssub), length(normRed));
    plot(tssub(1:min_len), normRed(1:min_len), 'color', 'r');
    xlabel('Time(s)','FontSize',14)
    ylabel('D2-RCAMP dF/F','FontSize',14)
    vline([Pickup_Seconds Down_Seconds Start_Seconds Stop_Seconds], {'k', 'k', 'g', 'g'});
    
    hold off
    

    folderName = 'dFF Figure';
    graphTitle = append(mouseNameAll, " ", folderName);
    save_figure(gcf, folderName, FIGURE_SAVE_DIR, graphTitle, SAVE_JPG, SAVE_EMF);
    
    if ~DISPLAY_FIGURES 
        close(gcf); 
    end
    
    %% Convert Rotarod key moments saved as seconds into frames elapsed
    % (Akira): Noise can cause rotarod & notepoint timestamps to be more 
    %          than 2 entries. In this case, Ellen advised me to pick the
    %          last two entries for each (which should correctly be the
    %          signals we want). Proceeding accordingly to this!
    Notepoints    = data.epocs.Note.onset * Fs - TAKEOUT_N_INITIAL_FRAMES;
    Rotarodpoints = data.epocs.PtAB.onset * Fs - TAKEOUT_N_INITIAL_FRAMES;
    %Notesoff     = data.epocs.Note.offset * Fs - 10000;
    
    Pickup = round(Notepoints(end-1,1));    % pickup should be the second last of notepoints
    Start  = round(Rotarodpoints(end-1,1)); % start the second last of rotarod
    Stop   = round(Rotarodpoints(end,1));   % stop the last of rotarod
    Down   = round(Notepoints(end,1));      % down the last of notepoints
    %End   = round(Notesoff(4,1));
    
    %% Calculate New z score using baseline and create plot
    % Uses 2.5 minutes before pickup as baseline
    BASELINE_EPOCH_SECOND_LENGTH = 150;
    BASELINE_EPOCH_FRAME_LENGTH = round(BASELINE_EPOCH_SECOND_LENGTH * Fs);

    Before_Green = normGreen(Pickup - BASELINE_EPOCH_FRAME_LENGTH:Pickup);
    stdev_Green = std(Before_Green);
    zscore_Green = normGreen/stdev_Green;
    
    Before_Red = normRed(Pickup - BASELINE_EPOCH_FRAME_LENGTH:Pickup);
    stdev_Red = std(Before_Red);
    zscore_Red = normRed/stdev_Red;
    
    if DISPLAY_FIGURES
        figure('units','normalized','outerposition',[0 0 1 1]);
    else % invisible if displayFigure is false
        figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
    end

    subplot(2,1,1);
    min_len = min(length(tssub), length(zscore_Green));
    plot(tssub(1:min_len), zscore_Green(1:min_len), 'color', [0 0.5 0]);
    xlabel('Time(s)','FontSize',14)
    ylabel('D1-GCAMP Z Score','FontSize',14)
    ylim([-5 8]);
    title(data.info.blockname(1,:),'Interpreter','none');
    vline([Pickup_Seconds Down_Seconds Start_Seconds Stop_Seconds], {'k', 'k', 'g', 'g'});
    
    hold on
    subplot(2,1,2);
    min_len = min(length(tssub), length(zscore_Red));
    plot(tssub(1:min_len), zscore_Red(1:min_len), 'color', 'r');
    xlabel('Time(s)','FontSize',14)
    ylim([-5 8]);
    ylabel('D2-RCAMP Z Score','FontSize',14)
    vline([Pickup_Seconds Down_Seconds Start_Seconds Stop_Seconds], {'k', 'k', 'g', 'g'});
    
    hold off
    
    folderName = 'Z Score Figure';
    graphTitle = append(mouseNameAll, " ", folderName);
    save_figure(gcf, folderName, FIGURE_SAVE_DIR, graphTitle, SAVE_JPG, SAVE_EMF);

    if ~DISPLAY_FIGURES 
        close(gcf); 
    end

    %% Plot new z score but setting beginning of experiment as reference
    tssub_start = tssub - Start_Seconds;
    note_start = data.epocs.Note.onset - Start_Seconds;
    rotarod_stop = Stop_Seconds - Start_Seconds;
    
    if DISPLAY_FIGURES
        figure('units','normalized','outerposition',[0 0 1 1]);
    else % invisible if displayFigure is false
        figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
    end

    subplot(2,1,1);
    min_len = min(length(tssub_start), length(zscore_Green));
    plot(tssub_start(1:min_len), zscore_Green(1:min_len), 'color', [0 0.5 0],'LineWidth',1.5);
    xlabel('Time(s)','FontSize',20,'FontWeight','bold');
    ylabel('D1-GCaMP6s Z Score','FontSize',20,'FontWeight','bold');
    ylim([-4 10]);
    xlim([note_start(1,1) - 150 note_start(2,1) + 150]);
    title(data.info.blockname(1,:),'Interpreter','none')
    vline([note_start(1,1) note_start(2,1) 0 rotarod_stop], {'k:', 'k:', 'k', 'k'});
    hold on
    subplot(2,1,2);
    min_len = min(length(tssub_start), length(zscore_Red));
    plot(tssub_start(1:min_len), zscore_Red(1:min_len), 'color', 'r','LineWidth',1.5);
    xlabel('Time(s)','FontSize',20,'FontWeight','bold');
    ylim([-4 10]);
    xlim([note_start(1,1) - 150 note_start(2,1) + 150]);
    ylabel('D2-RCaMP1b Z Score','FontSize',20,'FontWeight','bold');
    vline([note_start(1,1) note_start(2,1) 0 rotarod_stop], {'k:', 'k:', 'k', 'k'});
    
    hold off
    
    
    folderName = 'Z Score Start Align Figure';
    graphTitle = append(mouseNameAll, " ", folderName);
    save_figure(gcf, folderName, FIGURE_SAVE_DIR, graphTitle, SAVE_JPG, SAVE_EMF);

    if ~DISPLAY_FIGURES 
        close(gcf); 
    end
    
    %% Downsample data and save
    DOWNSAMPLE_BY = 1000;

    filt_zscore_Green = lowpassphotometry(zscore_Green,Fs,1);
    filt_zscore_Red   = lowpassphotometry(zscore_Red,Fs,1);

    zscore_down_Green = downsample(filt_zscore_Green, DOWNSAMPLE_BY)';
    zscore_down_Red   = downsample(filt_zscore_Red,   DOWNSAMPLE_BY)';
    tssub_start_down  = downsample(tssub_start,       DOWNSAMPLE_BY)';
    
    % also save an array holding data of:
    % - timestamps 
    % - z-score normalized & downsampled green signal
    % - z-score normalized & downsampled red signal
    % - key moments; note start, rotarod start & stop, note end
    startalign = NaN(300,4);
    startalign(1:length(tssub_start_down),  1) = tssub_start_down;
    startalign(1:length(zscore_down_Green), 2) = zscore_down_Green;
    startalign(1:length(zscore_down_Red),   3) = zscore_down_Red;
    startalign(1,4) = note_start(1,1);
    startalign(2,4) = 0;
    startalign(3,4) = rotarod_stop;
    startalign(4,4) = note_start(2,1);

    
    if SAVE_JPG || SAVE_EMF
        save_variable('startalign','Timestamps',FIGURE_SAVE_DIR, ...
                        mouseNameAll);
    end
    
    %% Calculate and save new Z score means
    if length(zscore_Green) < (Down + BASELINE_EPOCH_FRAME_LENGTH)
        fprintf("Shorter duration: This data lasts no longer than " + ...
             "%i frames after the mouse falls off the rotarod.\n" + ...
             "Will use available data to calculate signal means after " + ...
             "the mouse fell.\n", ...
             BASELINE_EPOCH_FRAME_LENGTH);
    end

    ZBefore_Green  = zscore_Green(Pickup - BASELINE_EPOCH_FRAME_LENGTH:Pickup);
    ZPickup_Green  = zscore_Green(Pickup:Start);
    ZRotarod_Green = zscore_Green(Start:Stop);
    ZPutdown_Green = zscore_Green(Stop:Down);
    ZAfter_Green   = zscore_Green(Down : min([length(zscore_Green), ...
                                         Down + BASELINE_EPOCH_FRAME_LENGTH]));
    
    Beforemean_Green  = mean(ZBefore_Green);
    Pickupmean_Green  = mean(ZPickup_Green);
    Trialmean_Green   = mean(ZRotarod_Green);
    Putdownmean_Green = mean(ZPutdown_Green);
    Aftermean_Green   = mean(ZAfter_Green);
    
    Means_Green = [Beforemean_Green Pickupmean_Green Trialmean_Green ...
                   Putdownmean_Green Aftermean_Green];
    % if DISPLAY_FIGURES figure; plot(Means_Green); end

    Meanshift_Green = Means_Green - Beforemean_Green;
    % if DISPLAY_FIGURES figure; plot(Meanshift_Green); end
    
    % RED AS WELL
    ZBefore_Red  = zscore_Red(Pickup-BASELINE_EPOCH_FRAME_LENGTH:Pickup);
    ZPickup_Red  = zscore_Red(Pickup:Start);
    ZRotarod_Red = zscore_Red(Start:Stop);
    ZPutdown_Red = zscore_Red(Stop:Down);
    ZAfter_Red   = zscore_Red(Down : min([length(zscore_Green), ...
                                     Down + BASELINE_EPOCH_FRAME_LENGTH]));
    
    Beforemean_Red  = mean(ZBefore_Red);
    Pickupmean_Red  = mean(ZPickup_Red);
    Trialmean_Red   = mean(ZRotarod_Red);
    Putdownmean_Red = mean(ZPutdown_Red);
    Aftermean_Red   = mean(ZAfter_Red);
    
    Means_Red = [Beforemean_Red Pickupmean_Red Trialmean_Red ...
                 Putdownmean_Red Aftermean_Red];
    % if DISPLAY_FIGURES figure; plot(Means_Red); end
    Meanshift_Red = Means_Red - Beforemean_Red;
    % if DISPLAY_FIGURES figure; plot(Meanshift_Red); end
    
    Means_All = [Means_Green; Meanshift_Green; Means_Red; Meanshift_Red];
    
    if SAVE_JPG || SAVE_EMF
        save_variable('Means_All','Means_All',FIGURE_SAVE_DIR, ...
                        mouseNameAll);
    end

    % compute other important info
    after_is_too_short = length(zscore_Green) < (Down + BASELINE_EPOCH_FRAME_LENGTH);
    note_onset_size = size(data.epocs.Note.onset);
    ptab_onset_size = size(data.epocs.PtAB.onset);

    % setting info about the length of time signals
    info = "";
    if length(green) ~= length(tssub)
        green_tssub_length = sprintf("length(green) %d ~= length(tssub) %d", ...
            length(green), length(tssub));
        disp(green_tssub_length);
        info = info + green_tssub_length;
    end
    if length(red) ~= length(tssub)
        red_tssub_length = sprintf("length(red) %d ~= length(tssub) %d", ...
            length(red), length(tssub));
        disp(red_tssub_length);
        info = info + red_tssub_length;
    end

    % save the result into resultData
    newData = struct( ...
        "Day", day, ...
        "Trial", trialInDay, ...
        "Means_Green",     Means_Green(1,3), ...
        "Meanshift_Green", Meanshift_Green(1,3), ...
        "Means_Red",       Means_Red(1,3), ...
        "Meanshift_Red",   Meanshift_Red(1,3),  ...
        "MouseName", mouseName, ...
        "TotalTrial", totalTrial, ...
        "After_TooShort", after_is_too_short, ...
        "Note_Onset_Size1", note_onset_size(1), ...
        "Note_Onset_Size2", note_onset_size(2), ...
        "PtAB_Onset_Size1", ptab_onset_size(1), ...
        "PtAB_Onset_Size2", ptab_onset_size(2), ...
        "Exception", '', ...
        "Info", info ...
        );
    resultData.adddata(mouseNameAll, newData);
    
    %%
    % Timestamps = [Pickup_Seconds Start_Seconds ...
    %               Stop_Seconds Down_Seconds];
    % if SAVE_JPG || SAVE_EMF
    %       save_variable('Timestamps','Timestamps',FIGURE_SAVE_DIR, ...
    %                     mouseNameTotalTrial);
    % end


    %% HELPER FUNCTIONS
    % Saves the given variable into the specified folder under given 
    %  saveDir with known graphTitle and in -ascii mode.
    % If saveFolder doesn't exist under saveDir, create it.
    function save_variable(variable, saveFolder, saveDir, graphTitle)
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

    % Saves the given figure into the specified folder under given 
    %  saveDir with known graphTitle.
    % If saveFolder doesn't exist under saveDir, create it.
    function save_figure(figure, saveFolder, saveDir, graphTitle, ...
            SAVE_JPG, SAVE_EMF)
        if ~SAVE_JPG && ~SAVE_EMF
            return
        end

        fullPathToFolder = fullfile(saveDir, saveFolder);
        % if saveFolder doesn't exist under saveDir, create it
        if ~exist(fullPathToFolder, 'dir')
            disp(append("Folder: ", fullPathToFolder, " did not " + ...
                "exist - created it!"))
            mkdir(fullPathToFolder)
        end
        % then save the content
        [~,name,~] = fileparts(graphTitle); graphTitle = name;
        figure_path_wo_ext = fullfile(fullPathToFolder, graphTitle);
        if SAVE_JPG
            saveas(figure, append(figure_path_wo_ext, '.jpg'));
        end
        if SAVE_EMF
            saveas(figure, append(figure_path_wo_ext, '.emf'));
        end
        
    end


end