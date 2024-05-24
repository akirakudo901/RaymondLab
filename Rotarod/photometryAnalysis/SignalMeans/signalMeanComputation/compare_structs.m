% Compare two structs

diff_structs(data, newData.data);

function diff_structs(s1, s2)
    internal(s1, s2, "BASE");

    function internal(s1, s2, soFar)
        fields1 = convertCharsToStrings(sort(fieldnames(s1)))';
        fields2 = convertCharsToStrings(sort(fieldnames(s2)))';
    
        if (length(fields1) ~= length(fields2)) || ...
           (~prod([fields1(1:end)] == [fields2(1:end)]))
            fprintf("Fields are different at %s for %s and %s.\n", ...
                soFar, fields1, fields2);
            return
        end
    
        for f = fields1
            if isstruct(s1.(f))
                internal(s1.(f), s2.(f), append(soFar, ".", f))
            else
                val1 = s1.(f); val2 = s2.(f);
                if iscell(val1)
                    val1 = val1{1}; val2 = val2{1};
                end
    
                if ~prod(isequal(val1, val2))
                    fprintf("Entries are different at %s, %s and %s.\n", ...
                        append(soFar, ".", f), val1, val2)
                end
            end
        end
    end

end