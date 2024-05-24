function data = lowpassphotometry(data, sampling_rate, bandwidth)
%LOWPASS
%Used to filter normalized data. Eliminates noise (high frequency peaks). 
order=2;
[b,a] = butter(order,[bandwidth(1)]/(sampling_rate/2),'low');
data=filtfilt(b,a,data);
end