
function [normDat] = deltaFF (dat1, dat2)

reg = polyfit(dat2, dat1, 1); 
a = reg(1);
b = reg(2);
controlFit = a.*dat2 + b;

normDat = (dat1 - controlFit)./ controlFit; %this gives deltaF/F
normDat = normDat * 100;
end 