% A sanity check for whether ResultData works
CSV_DIR = fullfile("test", "csvs");

sanity_check_ResultData(CSV_DIR)

%% Functions
function sanity_check_ResultData(CSV_DIR)
    rd = ResultData(["Color", "Price", "StillUsable"]);

    rd.loaddata(fullfile(CSV_DIR, "test_ResultData_before.csv"));
    
    for i = 1:5
        newData = struct("Color", append("Green ", int2str(i)), ...
            "Price", i*100, ...
            "StillUsable", (mod(i, 2) == 0) ...
            );
        rd.adddata(append("Color", int2str(i)), newData);
    end

    assert(rd.iskey("Color1"));
    assert(rd.iskey("Color2"));
    assert(rd.iskey("Color3"));
    assert(rd.iskey("Color4"));
    assert(rd.iskey("Color5"));
    assert(~rd.iskey("THIS SHOULDN'T BE A KEY"));
    
    rd.savedata(CSV_DIR, "test_ResultData_after.csv");
end