
outer(1, 10, 1, false)

function inner_func(start, finish, jump, printOdd)
    for i = start:jump:finish
        if printOdd && mod(i, 2) == 1
            disp(i)
        elseif mod(i, 2) == 0
            disp(i)
        end
    end
end

function outer(start, varargin)
    inner_func(start, varargin{:})
end