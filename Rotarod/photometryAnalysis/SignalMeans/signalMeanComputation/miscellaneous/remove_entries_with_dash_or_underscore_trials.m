FILENAME_IDX = 1; DAY_IDX = 2; TRIAL_IDX = 3;

FILENAME = "ALLDATA_NOERROR_NOSPECIALFILES_photometry_means_analysis_result.csv";
FOLDERPATH = fullfile(".", "results");
FILEPATH = fullfile(FOLDERPATH, FILENAME);

data = readcell(FILEPATH);

% first find all entries for which the name includes a "-" or "_"
trials = data(:, TRIAL_IDX);
nameIncludesDashOrUnderscore = @(x) (~isnumeric(x) && ...
    (contains(x, "_") || contains(x, "-")));
rowsToRemove = cellfun(nameIncludesDashOrUnderscore, trials);
newData = data(rowsToRemove, :);
removedData = data(~rowsToRemove, :);

% save result
writecell(newData, FILEPATH);