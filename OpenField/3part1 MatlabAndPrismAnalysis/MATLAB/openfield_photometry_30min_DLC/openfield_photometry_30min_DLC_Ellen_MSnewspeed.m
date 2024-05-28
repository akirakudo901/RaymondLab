clc
clear
close all

%% Get file and plot mouse position over time using TailBase position
% note: these bodyparts are in different columns in the csv with BSOID
% labels
% 30 minutes total
% The video is analyzed from 30 seconds (0:30) to 30:30 not necessarily
% If file is shorter than that, to end of video but make sure you note this
% Change TailBasex/TailBasey to the x and y you are interested in

file = uigetfile('.csv');
data = readmatrix(file);
graphtitle = file(1:9); % fix this because not all files are consistent
%% get rid of positions based on dlc probability

%Position of Tail Base
TailBasex = data(:,18); % fix this
TailBasey = data(:,19);
for i=1:length(data(:,20))
    if data(i,20) <0.95
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
    if data(i,22) <0.95
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
    if data(i,4) <0.95
        Snoutx(i,1) = NaN;
        Snouty(i,1) = NaN;
    end
end

Snoutx = repnan(Snoutx,'linear');
Snouty = repnan(Snouty,'linear');

Time = data(:,1)/40;
    Time = Time(1200:72000);% Time(1200:73200);originaly but file short
    TailBasex = TailBasex(1200:72000);
    TailBasey = TailBasey(1200:72000);
    Bellyx = Bellyx(1200:72000);
    Bellyy = Bellyy(1200:72000);
    Snoutx = Snoutx(1200:72000);
    Snouty = Snouty(1200:72000);

% Plot and save TailBase positions
% Change path for saveas to your computer's path
figure; plot(TailBasex, TailBasey);
hold on
title(graphtitle,'Interpreter','none')
hold off
%saveas(gcf,['Z:\Raymond Lab\Leland\Open Field Analysis\Position Graphs\',graphtitle,'.jpg']);

% Plot and save Belly positions
% Change path for saveas to your computer's path
figure; plot(Bellyx, Bellyy);
hold on
title(graphtitle,'Interpreter','none')
hold off
%saveas(gcf,['Z:/Raymond Lab/Leland/Open Field Analysis/Position Graphs/',graphtitle,'.jpg']);

% Plot and save Snout positions
% Change path for saveas to your computer's path
figure; plot(Snoutx, Snouty);



%% Calculate total distance travelled
% Using tailbase position

xdiff = diff(TailBasex);
ydiff = diff(TailBasey);
totaldiff = sqrt(xdiff.^2 + ydiff.^2);
totaldistance = sum(totaldiff,'omitnan');

%% moving outliers only
movediff = totaldiff > sqrt(2); % moving more than 2 pixels
amovediff = totaldiff(movediff); 
meanmove = mean(amovediff);
outliers = isoutlier(amovediff,'mean');
omovediff = amovediff(outliers);
minout = min(omovediff);
for i=1: length (totaldiff)
    if totaldiff(i) > (minout)
        totaldiff(i) = NaN;
    end
end

mtotaldiff = repnan (totaldiff,'linear');


%% Calculate ranges that represent the full size of the open field, convert ranges to cm
% Calculate total distance travelled in cm
xrange = max(Snoutx) - min(Snoutx);
yrange = max(Snouty) - min(Snouty);
totalrange = sqrt(xrange.^2 + yrange.^2);
x_cm = 38/xrange;
y_cm = 38/yrange;

total_cm = 53.7401153702/totalrange;

totaldiff_cm = abs(mtotaldiff*total_cm);
totaldistance_cm = sum(totaldiff_cm,'omitnan');

disp 'total distance in cm is';
disp(totaldistance_cm);
mdistance = mean (totaldiff_cm);


%% Calculate total distance per 5 minutes
 
pixelDistanceOne = nansum(totaldiff(1:11999));
pixelDistanceTwo = nansum(totaldiff(12000:23999));
pixelDistanceThree = nansum(totaldiff(24000:35999));
pixelDistanceFour = nansum(totaldiff(36000:47999));
pixelDistanceFive = nansum(totaldiff(48000:59999));
pixelDistanceSix = nansum(totaldiff(60000:71999));
% pixelDistanceSeven = nansum(totaldiff(72000:83999));
% pixelDistanceEight = nansum(totaldiff(84000:95999));
% pixelDistanceNine = nansum(totaldiff(96000:107999));
% pixelDistanceTen = nansum(totaldiff(108000:119999));
% pixelDistanceEleven = nansum(totaldiff(120000:131999));
% pixelDistanceTwelve = nansum(totaldiff(132000:end));
  
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
 
%if(abs(totalDistance - totaldistance_cm) > 0.01)
    disp("WARNING: Distances don't match; Values not saved");
%else
    save(['Z:/Raymond Lab/Leland/Open Field Analysis/Total distance per interval/', graphtitle],'distanceByintervals', '-ascii');
%end

%% Calculate mouse movement and speed
% Based on 40 Hz frame rate
% note add code to measure speed during bouts of motion
speed = abs(totaldiff_cm/0.025);

averagespeed = mean(speed,'omitnan');
disp('average speed is');
disp(averagespeed);

Timenew = Time(2:end);
speedfigure = figure; plot(Timenew,speed);
