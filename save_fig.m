function save_fig(name, w, h)
% Save the current figure as PNG (w x h pixels). Works in Octave and MATLAB.
    if exist('OCTAVE_VERSION', 'builtin')
        print(gcf, name, '-dpng', sprintf('-S%d,%d', w, h));
    else
        set(gcf, 'PaperPositionMode', 'auto', 'Position', [100 100 w h]);
        print(gcf, name, '-dpng', '-r100');
    end
end
