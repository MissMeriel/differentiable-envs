function plotHeatScatter(y,x,yname,xname)
    [counts, xedge,yedge,binx,biny] = histcounts2(x,y,100);
    %imagesc(counts,'XData',xedge,'YData',yedge)
    hold on
    hasPoints = find(counts(:)>0);
    
    c = colormap();
    max_log = floor(log10(max(counts(:))));
    counts = log10(counts);

    
    countmap = linspace(min(counts(hasPoints)),max(counts(hasPoints)),size(c,1));
    cbh = colorbar;
    tickvalues = interp1(countmap,linspace(0,1,size(c,1)),0:max_log);
    cbh.Ticks = tickvalues(~isnan(tickvalues));
    tl = splitlines(sprintf('10^%i\n',0:max_log))';
    %tl = tl(1:max_log);
    cbh.TickLabels = tl;
    for ind = hasPoints'
        [i,j] = ind2sub(size(counts),ind);
        pointsInThisBin = binx==i & biny ==j;
        pointsInThisBin = find(pointsInThisBin,1000);
        plot(x(pointsInThisBin),y(pointsInThisBin),'.',color=interp1(countmap,colormap,counts(i,j)),HandleVisibility='off')
    end
    % mdl = fitlm(x,y,'RobustOpts','on')
    % output = mdl.predict([0;max(x)])';
    % handle = plot([0,max(x)],[output],'k');
    xlabel(xname,'Interpreter','none')
    ylabel(yname,'Interpreter','none')
    
    
end