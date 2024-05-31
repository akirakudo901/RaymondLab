%% A result-data class which stores all results from a set of 
% analysis without file name duplicates, and allows saving and loading 
% of the result into a single csv file.

% This consists of: 
%  - a set, made out of map, tracking which file has
%    been created, 
%  which maps each data to:
%  - structures for the data fields of that file

classdef ResultData < handle
    properties
        header string;
    end
    properties (SetAccess = private)
        dataMap containers.Map
    end
    methods
       % initializes this ResultData header with fields & an empty map
       function obj = ResultData(fields)
           obj.header = strings([1, length(fields) + 1]);
           obj.header(1,1) = "fileName";
           obj.header(1,2:end) = fields(1:end);
           obj.dataMap = containers.Map;
       end

       % returns true if a data with given name is already stored
       function dataIsKey = iskey(obj, name)
           nameStr = string(name);
           dataIsKey = isKey(obj.dataMap, nameStr);
       end

       % get value by giving a key
       function value = get(obj, key)
           keyStr = string(key);
           value = obj.dataMap(keyStr);
       end
       
       % get all keys stored in this data storage
       function allKeys = getkeys(obj)
           allKeys = keys(obj.dataMap);
       end

       % get all values stored in this storage based on specific key order
       function allValues = getvalues(obj, keySet)
           allValues = values(obj.dataMap, keySet);
       end

       % adds a new data entry with new data passed as struct
       function adddata(obj, name, newData)
           % ensure newData contains fields matching obj.header
           % can contain more fields, which are ignored
           if ~all(isfield(newData, obj.header(2:end)))
               return
           end
           % insert the new data
           obj.dataMap(name) = newData;
       end

       % remove data entry by name
       function removedatabyname(obj, name)
           remove(obj.dataMap, name);
       end

       % save data into the given file path
       function savedata(obj, saveDir, fileName)
           disp(append("Save ResultData data as fileName: ", fileName, ...
                       "; under directory: ", saveDir));
           fullFilePath = fullfile(saveDir, fileName);

           % convert Map into cell array; Map is of the form 
           % fileName : valArray
       
           mapKeys = keys(obj.dataMap);
           mapValues = values(obj.dataMap, mapKeys);
           % preallocate variable space
           savedCellArray = cell(length(mapKeys), length(obj.header));
           % add the header
           savedCellArray(1, :) = num2cell(obj.header);

           % for every index in key array, where keys are file ids
           for i = 1:length(mapKeys)                
               k = mapKeys{i}; v = mapValues{i};
               % insert the data corresponding to files (the values)
               % name first
               savedCellArray{i + 1, 1} = k;
               % then the values characterizing the data point
               for idx = 2:length(obj.header)
                   fieldName = obj.header(idx);
                   savedCellArray{i + 1, idx} = v.(fieldName);
               end
           end

           % then save the cell array into a CSV file
           writecell(savedCellArray,fullFilePath);

           disp(append("Saving of: ", fileName, "; successful!"));
       end

       % load data from the given file path
       function loaddata(obj, filePath)
           disp(append("Load ResultData from path: ", filePath));
           % read and load data from file into a cell array
           dataCellArray = readcell(filePath, "NumHeaderLines", 0, ...
               "Delimiter",",");

           dataCellArray = cellfun(@replace_missing_with_empty_char, ...
               dataCellArray, 'UniformOutput',false);

           % first parse header to check it matches our expected one
           % if not, raise error
           actualHeader = dataCellArray(1,:);
           % we remove empty entries in the header
           nonCharMissing = @(charArr) ismissing(convertCharsToStrings(charArr));
           noMissingHeader = actualHeader(~cellfun(nonCharMissing, actualHeader));
           % for every entry position in actualHeader
           for i=1:length(noMissingHeader)
               % if the header entry doesn't match what's expected, error
               if ~strcmp(convertCharsToStrings(noMissingHeader{i}), ...
                          obj.header{i})
                   ME = MException('ResultData:headerNotMatching', ...
                        'Header of loaded table is different from expected - aborting.');
                   throw(ME);
               end
           end

           % once we know the header is as expected
           % for each row (file) in dataCellArray, get name & fields
           % then add new loaded data into map
           [numRow,numFields] = size(dataCellArray);
           for rowIdx = 2:numRow
               % obtain the new data name & structure
               name = dataCellArray{rowIdx,1};
               newData = struct;
               for idx = 2:length(obj.header) % all fields ~= fileName
                   fieldName = obj.header(idx);
                   % if index is beyond cell array range, we are adding a
                   % new field
                   if idx <= numFields
                       newData.(fieldName) = dataCellArray{rowIdx, idx};
                   else
                       newData.(fieldName) = '';
                   end
               end
               % add the new data
               obj.adddata(name, newData);
           end
           
           disp(append("Successful loading of: ", filePath, "!"));
           disp("");

           function returned = replace_missing_with_empty_char(cellEntry)
                if ismissing(cellEntry)
                    returned = '';
                    return
                else
                    returned = cellEntry;
                    return
                end
           end
       end

       % save data into csv format such that prism can be linked to it
%        function savedataforprism(obj, saveDir, commonName)
%            disp(append("Saving ResultData data for prism as files: ", ...
%                commonName, "; under directory: ", saveDir));
% 
%            % convert Map into cell array; Map is of the form 
%            % fileName : valArray
%        
%            mapKeys = keys(obj.dataMap);
%            mapValues = values(obj.dataMap, mapKeys);
% 
%            % - totalDistanceCm, centerTime, 
%            %   timeFractionByQuadrant, distanceByIntervals
% 
%            % Allocate such that two columns, each headered by mouseType,
%            % are populated by individual file values, with row titles
%            % corresponding to file names.
%            % header: {fileName, mouseType1, mouseType2, ...}
%            
%            % preallocate variable space - likely have >= 1 mice types
%            type1CellArrays = repelem({cell(length(mapKeys), 2)}, 4);
%            % add a simple header - ensuring inserted mouseTypes are strings
%            mT1 = mapValues{1}.mouseType;
%            if ischar(mT1)
%                mT1 = convertCharsToStrings(mT1);
%            end
%            % add the header to all cell arrays
%            for cAIdx = 1:length(type1CellArrays)
%                type1CellArrays{1,cAIdx}(1,:) = {"fileName", mT1};
%            end
%            
%            % for every index in key array, where keys are file ids
%            for fileId = 1:length(mapKeys)                
%                k = mapKeys{fileId}; v = mapValues{fileId};
%                % insert the data corresponding to files (the values)
%                % this type1CellValues should match type1CellNames
%                type1CellValues = {v.totalDistanceCm, ... 
%                                   v.centerTime, ...
%                                   v.timeFractionByQuadrant, ...
%                                   v.distanceByIntervals };
% 
%                % get index of mouseType
%                mouseTypeIdx = find(cellfun( ...
%                    @(x) strcmp(x, v.mouseType), ...
%                    type1CellArrays{1,1}(1,:)), ...
%                    1);
%                % argument 1 limit search to 1 entry and makes it fast
% 
%                % for each cell array
%                for cAIdx = 1:length(type1CellArrays)
%                    % insert name first
%                    type1CellArrays{1,cAIdx}{fileId + 1,1} = k;
%                    % then the values characterizing the data point
%                    % if mouseType not listed, we add a new column for it
%                    if isempty(mouseTypeIdx)
%                        type1CellArrays{1,cAIdx}{1,end + 1} = convertCharsToStrings(v.mouseType);
%                        newMouseTypeIdx = length(type1CellArrays{1,cAIdx}(1,:));
%                    else
%                        newMouseTypeIdx = mouseTypeIdx;
%                    end
%                    % then add value to the corresponding location
%                    type1CellArrays{1, cAIdx}{fileId + 1, newMouseTypeIdx} = ...
%                        type1CellValues{cAIdx};
%                end
%                ...
%            end
%            ...
% 
%            % then save the cell arrays
%            % write the cell array into a CSV file
%            % this type1CellNames should match type1CellValues
%            type1CellNames = {'totalDistanceCm', ... 
%                              'centerTime', ...
%                              'timeFractionByQuadrant', ...
%                              'distanceByIntervals' };
%            for i = 1:length(type1CellArrays)
%                % get full name from type1CellNames
%                fileName = append(commonName, "_", type1CellNames{i}, ".csv");
%                fullFilePath = fullfile(saveDir, fileName);
%                writecell(type1CellArrays{i},fullFilePath);
%                
%                disp(append("Saving of: ", fileName, "; successful!"));
%            end
% 
%            % TODO SAVE SIMILAR VALUES FOR OVER TIME
%        ...
%        end
   ...
   end
...
end