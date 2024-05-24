
GenotypePath = "C:\Users\mashi\Desktop\RaymondLab\Experiments\Rotarod\photometryAnalysis\SignalMeans\Akira_photometry_rotarod_scripts\Ellen script\Akira_Kudo_COPY 2-Colour Photometry Experiments - CSV_Genotype Per Mice.csv";

ChangeContentPath = "C:\Users\mashi\Desktop\RaymondLab\Experiments\Rotarod\photometryAnalysis\SignalMeans\Akira_photometry_rotarod_scripts\Ellen script\results\ALL_BUT_SPECIAL_FILES_photometry_means_analysis_result.csv";

G_MOUSENAME_IDX = 1; G_GENOTYPE_IDX = 2;

genotypeArray = readcell(GenotypePath, "NumHeaderLines", 0, ...
               "Delimiter",",");

C_FILENAME_IDX = 1; C_MOUSENAME_IDX = 8; 

changeArray = readcell(ChangeContentPath, "NumHeaderLines", 0, ...
               "Delimiter",",");

changeArray{1, end + 1} = "Genotype";

[g_r,~] = size(genotypeArray);
[c_r,changeArrayColumnSize] = size(changeArray);

for idx = 2:g_r
    % mouse name: 'cageNum mfNum'
    mouseName = string( genotypeArray{idx, G_MOUSENAME_IDX} );
    % find space to divide mouse name to cage number and mnNum
    parts = strsplit(mouseName, " ");
    if length(parts) ~= 2 % skip if pattern is not matched
        fprintf("Mouse name %s does not match the name pattern.\n", mouseName);
        continue
    end
    g_cageNum = parts(1); g_mnNum = parts(2);

    for j = 2:c_r
        % full mouse name: cageNum_mfNum_trialNum
        mouseName = changeArray{j, C_FILENAME_IDX};
        if contains(mouseName, "Failed_") continue; end
        % mouse name without trial: cageNum_mfNum
        mouseNameInChangedArray = mouseName(1:end-3);
        % extract cageNum and mnNum
        c_cageNum = regexp(mouseNameInChangedArray, "\d+", 'match');
        c_cageNum = c_cageNum{1};
        c_mnNum = regexp(mouseNameInChangedArray, "[mf]\d+", 'match');
        c_mnNum = c_mnNum{1};
        % if c_cageNum is contained in g_cageNum and mnNum are matching
        % reason why we check for containment is because some c_cageNum 
        % are abbreviateions of g_cageNum, likely
        if contains(g_cageNum, c_cageNum) && strcmp(g_mnNum, c_mnNum)
            % add entry of genotype
            changeArray{j, changeArrayColumnSize} = genotypeArray{idx, G_GENOTYPE_IDX};
        end
    end
end

changeArray = cellfun(@replaceIsMissing, changeArray, UniformOutput=false);

writecell(changeArray, ChangeContentPath);

function returned = replaceIsMissing(missing)
    if ismissing(missing)
        returned = ""; return
    else
        returned = missing;
    end
end