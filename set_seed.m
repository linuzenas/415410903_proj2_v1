function set_seed(s)
% Make the random numbers repeatable (works in Octave and in MATLAB).
    if exist('OCTAVE_VERSION', 'builtin')
        rand('state', s); randn('state', s);
    else
        rand('state', s); randn('state', s);   % legacy generator, same call as in MATLAB
    end
end
