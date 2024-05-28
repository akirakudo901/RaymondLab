clc
clear all
close all

%% Get file and plot mouse position over time using TailBase position
% 30 minutes total
% The video is analyzed from 30 seconds (0:30) to 30:30
% If file is shorter than that, to end of video but make sure you note this
% Change TailBasex/TailBasey to the x and y you are interested in

% Constant pointing to directory to which we save all data
% Change path to your computer's path
% save_dir = fullfile('X:\','Raymond Lab', ...
%            '2 Colour D1 D2 Photometry Project', ...
%            'Open Field', 'Open Field Data Analysis');
% save_dir = "315955_m1";
save_dir = "326787_m2";


file = uigetfile('.csv');
data = readmatrix(file);
% graphtitle = "315955_m1";
graphtitle = "326787_m2";


%Position of Tail Base
TailBasex = data(:,17);
TailBasey = data(:,18);
for i=1:length(data(:,19))
    if data(i,19) <0.8
        TailBasex(i,1) = NaN;
        TailBasey(i,1) = NaN;
    end
end

TailBasex = repnan(TailBasex,'linear');
TailBasey = repnan(TailBasey,'linear');

%Position of Belly
Bellyx = data(:,20);
Bellyy = data(:,21);
for i=1:length(data(:,22))
    if data(i,22) <0.8
        Bellyx(i,1) = NaN;
        Bellyy(i,1) = NaN;
    end
end

Bellyx = repnan(Bellyx,'linear');
Bellyy = repnan(Bellyy,'linear');

%Position of Snout
Snoutx = data(:,2);
Snouty = data(:,3);
for i=1:length(data(:,4))
    if data(i,4) <0.8
        Snoutx(i,1) = NaN;
        Snouty(i,1) = NaN;
    end
end

Snoutx = repnan(Snoutx,'linear');
Snouty = repnan(Snouty,'linear');

TIME_START = 1200;
TIME_END = 73023; %73200;

Time = data(:,1)/40;
    Time =           Time(TIME_START:TIME_END);
    TailBasex = TailBasex(TIME_START:TIME_END);
    TailBasey = TailBasey(TIME_START:TIME_END);
    Bellyx =       Bellyx(TIME_START:TIME_END);
    Bellyy =       Bellyy(TIME_START:TIME_END);
    Snoutx =       Snoutx(TIME_START:TIME_END);
    Snouty =       Snouty(TIME_START:TIME_END);



tailBasePlotName = append(graphtitle, ' Tail Base Coordinates');
bellyPlotName    = append(graphtitle, ' Belly Coordinates');
snoutPlotName    = append(graphtitle, ' Snout Coordinates');

% Plot and save TailBase positions
figure; plot(TailBasex, TailBasey);
hold on
title(tailBasePlotName,'Interpreter','none')
hold off
saveas(gcf,fullfile(save_dir,'/Position Graphs', ...
    append(tailBasePlotName,'.png')));

% Plot and save Belly positions
figure; plot(Bellyx, Bellyy);
hold on
title(bellyPlotName,'Interpreter','none')
hold off
saveas(gcf,fullfile(save_dir,'/Position Graphs', ...
    append(bellyPlotName,'.png')));

% Plot and save Snout positions
figure; plot(Snoutx, Snouty);
hold on
title(snoutPlotName,'Interpreter','none')
hold off
saveas(gcf,fullfile(save_dir,'/Position Graphs', ...
    append(snoutPlotName,'.png')));



%% Calculate total distance travelled
% Using tailbase position

xdiff = diff(TailBasex);
ydiff = diff(TailBasey);
totaldiff = sqrt(xdiff.^2 + ydiff.^2);
%outliers = isoutlier(totaldiff,'mean');
%totaldiff = filloutliers(totaldiff,'linear');
%totaldiff = repnan(totaldiff,'linear');
totaldistance = sum(totaldiff,'omitnan');
%totaldistancepartial = sum(totaldiff(end-11400:end),'omitnan');
disp 'total distance in pixels is'
disp(totaldistance);
save( fullfile(save_dir, 'Total distance pixels', graphtitle), ...
    'totaldistance', '-ascii');




%% Calculate ranges, convert ranges to cm
% Calculate total distance travelled in cm
xrange = max(Snoutx) - min(Snoutx);
yrange = max(Snouty) - min(Snouty);
totalrange = sqrt(xrange.^2 + yrange.^2);
x_cm = 38/xrange;
y_cm = 38/yrange;

total_cm = 53.7401153702/totalrange;

xdiff_cm = xdiff*x_cm;
ydiff_cm = ydiff*y_cm;
totaldiff_cm = abs(totaldiff*total_cm);
totaldistance_cm = sum(totaldiff_cm);
disp 'total distance in cm is';
disp(totaldistance_cm);

save( fullfile(save_dir, 'Total diff cm', graphtitle), ...
    'totaldiff_cm', '-ascii');
save( fullfile(save_dir, 'Total distance cm', graphtitle), ...
    'totaldistance_cm', '-ascii');
save( fullfile(save_dir, 'Total distance cm partial', graphtitle), ...
    'totaldistance_cm', '-ascii');
%% Calculate center time
% Convert pixels to cm

Snoutx_cm = Snoutx*x_cm;
Snouty_cm = Snouty*y_cm;
Bellyx_cm = Bellyx*x_cm;
Bellyy_cm = Bellyy*y_cm;
 
xcentermin = (min(Snoutx_cm)+8);
xcentermax = (max(Snoutx_cm)-8);
ycentermin = (min(Snouty_cm)+8);
ycentermax = (max(Snouty_cm)-8);
 
Bellyx_cm(Bellyx_cm < xcentermin) = NaN;
Bellyx_cm(Bellyx_cm > xcentermax) = NaN;
Bellyy_cm(Bellyy_cm < ycentermin) = NaN;
Bellyy_cm(Bellyy_cm > ycentermax) = NaN;
Bellyxy_cm = [Bellyx_cm, Bellyy_cm];
 
Centertime = (Bellyxy_cm(:,1) + Bellyxy_cm(:,2));
Centertime = (Centertime > 0);
Centertimesum = (sum(Centertime)/(length(Time))*100);

save( fullfile(save_dir, 'Center time', graphtitle), ...
    'Centertimesum', '-ascii');

disp('Center time is');
disp(Centertimesum);

%% Calculate time spent in each of 4 quadrants
% Quadrants increment clockwise starting with the bottom left quadrant (Quad1) 

xmid = min(Snoutx) + xrange/2;
ymid = min(Snouty) + yrange/2;

Quad1 = 0;
Quad2 = 0;
Quad3 = 0;
Quad4 = 0;

for i = 1 : length(Bellyx)
    if (Bellyx(i) < xmid) && (Bellyy(i) <= ymid)
        Quad1 = Quad1 + 1;
    elseif (Bellyx(i) <= xmid) && (Bellyy(i) > ymid)
        Quad2 = Quad2 + 1;
    elseif (Bellyx(i) > xmid) && (Bellyy(i) >= ymid)
        Quad3 = Quad3 + 1;
    else
        Quad4 = Quad4 + 1;
    end
end

timeQuad1 = Quad1/40;
timeQuad2 = Quad2/40;
timeQuad3 = Quad3/40;
timeQuad4 = Quad4/40;

timeByQuad = [timeQuad1, timeQuad2, timeQuad3, timeQuad4];
sumOfTime = sum(timeByQuad);

fracInQuad1 = (timeQuad1 / sumOfTime) * 100;
fracInQuad2 = (timeQuad2 / sumOfTime) * 100;
fracInQuad3 = (timeQuad3 / sumOfTime) * 100;
fracInQuad4 = (timeQuad4 / sumOfTime) * 100;

timeByQuadFrac = [fracInQuad1, fracInQuad2, fracInQuad3, fracInQuad4];

if sumOfTime - length(Bellyx)/40 > 0.1
    disp("WARNING: Times don't add to 60 minutes; Values not saved");
end
    save( fullfile(save_dir, 'Time by quadrants', graphtitle), ...
        'timeByQuad', '-ascii');
    save( fullfile(save_dir, 'Fraction of time by quadrants', graphtitle), ...
        'timeByQuadFrac', '-ascii');


%% Calculate total distance per 5 minutes
INTERVAL_SIX_START = 60000;
INTERVAL_SIX_END = 71823; %71823;

pixelDistanceOne = sum(totaldiff(1:11999),'omitnan');
pixelDistanceTwo = sum(totaldiff(12000:23999),'omitnan');
pixelDistanceThree = sum(totaldiff(24000:35999),'omitnan');
pixelDistanceFour = sum(totaldiff(36000:47999),'omitnan');
pixelDistanceFive = sum(totaldiff(48000:59999),'omitnan');
pixelDistanceSix = sum(totaldiff(INTERVAL_SIX_START:INTERVAL_SIX_END),'omitnan');
% pixelDistanceSeven = sum(totaldiff(72000:83999),'omitnan');
% pixelDistanceEight = sum(totaldiff(84000:95999),'omitnan');
% pixelDistanceNine = sum(totaldiff(96000:107999),'omitnan');
% pixelDistanceTen = sum(totaldiff(108000:119999),'omitnan');
% pixelDistanceEleven = sum(totaldiff(120000:131999,'omitnan'));
% pixelDistanceTwelve = sum(totaldiff(132000:end,'omitnan'));
  
cmDistanceOne = abs(pixelDistanceOne * total_cm);
cmDistanceTwo = abs(pixelDistanceTwo * total_cm);
cmDistanceThree = abs(pixelDistanceThree * total_cm);
cmDistanceFour = abs(pixelDistanceFour * total_cm);
cmDistanceFive = abs(pixelDistanceFive * total_cm);
cmDistanceSix = abs(pixelDistanceSix * total_cm);
% cmDistanceSeven = abs(pixelDistanceSeven * total_cm);
% cmDistanceEight = abs(pixelDistanceEight * total_cm);
% cmDistanceNine = abs(pixelDistanceNine * total_cm);
% cmDistanceTen = abs(pixelDistanceTen * total_cm);
% cmDistanceEleven = abs(pixelDistanceEleven * total_cm);
% cmDistanceTwelve = abs(pixelDistanceTwelve * total_cm);
 
distanceByintervals = [cmDistanceOne, cmDistanceTwo, cmDistanceThree, cmDistanceFour, cmDistanceFive, cmDistanceSix];
totalDistance = sum(distanceByintervals);
 
if(abs(totalDistance - totaldistance_cm) > 0.01)
    disp("WARNING: Distances don't match; Values not saved");
else
    save( fullfile(save_dir, 'Total distance per interval', graphtitle), ...
        'distanceByintervals', '-ascii');
end



%% Calculate mouse movement and speed
% Removes impossible speeds by filling outliers
% Based on 40 Hz frame rate

%xspeed = abs(xdiffinches/0.025);
%yspeed = abs(ydiffinches/0.025);
totalspeed = abs(totaldiff_cm/0.025);

averagespeed = mean(totalspeed,'omitnan');
disp('average speed is');
disp(averagespeed);

Timenew = Time(2:end);
speedfigure = figure; plot(Timenew,totalspeed);
%daspect([1 0.01 1])
saveas(gcf,fullfile(save_dir,'/Speed plots',append(graphtitle,'.png')));
%timemoving = (totalspeed >= 3);

save( fullfile(save_dir, 'Speed over time', graphtitle), ...
    'totalspeed', '-ascii');
save( fullfile(save_dir, 'Average speed', graphtitle), ...
    'averagespeed', '-ascii');

%% Data to copy to Prism

Data_To_Copy = zeros(12, 4);
Data_To_Copy(1,1) = totaldistance_cm;
Data_To_Copy(1,2) = Centertimesum;
Data_To_Copy(1:4,3) = timeByQuadFrac; 
Data_To_Copy(1:6,4) = distanceByintervals;
openvar('Data_To_Copy');