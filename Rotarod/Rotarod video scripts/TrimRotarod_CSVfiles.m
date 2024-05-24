%% Import data from TDT files
% Ensure you are IN the folder for the trial for this part
clear all
    
%%
data = TDTbin2mat(uigetdir);


%% Extract rotarod start and stop times
% Video is recorded Fs (40) frames per second. Extract rotarod start and 
% rotarod end time and convert to frames
% These times are extracted either from the photometry data file
Fs = 40;

Rotarod_Times = data.epocs.PtAB.onset;

%% If there are more than 2 rotarod times, you must edit the Rotarod_Times variable to have the correct start and stop time
% If you're not sure, please check the video or the Prism document or ask
    
if length(Rotarod_Times) ~= 2
    disp('check Rotarod_Times and edit as needed');
    openvar('Rotarod_Times')
    return
else
end

%% Rotarod Times are changed to not include the last 2 seconds to account for rotations

Rotarod_Times(2,1) = Rotarod_Times(2,1) - 2; 

%% Extract frames for trimming .csv files

Rotarod_Frames = round(Rotarod_Times*40)

% Use these frames to trim the DLC .csv files

%% Save data into current folder (make sure this is the correct photometry folder)

save("Rotarod_Times.mat","Rotarod_Times");
save("Rotarod_Frames.mat","Rotarod_Frames");