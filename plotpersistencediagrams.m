function [] = plotpersistencediagrams( PDs )
    PP = cell2mat(PDs');
    M=max(PP(:));
    m=min(PP(:));
    M=max(M,0);
    m=min(m,0);  % Ensure that the plot includes the origin.

    diagonal=linspace(1.2*m,1.2*M,2);
    hold on

    axis([1.2*m 1.2*M 1.2*m 1.2*M]); % set axes to include all points, with a bit of space on both sides

    strs = {};
    for ii = 1:length(PDs)
        I = PDs{ii};
        plot(I(:, 1), I(:, 2),'.', 'markers', 12); % plot the points on the persistence diagram
        strs{end+1} = sprintf('H%i', ii-1);
    end
    xlabel('Birth Time');
    ylabel('Death Time');
    legend(strs);
    plot(diagonal,diagonal, 'k');  % plot the diagonal
    hold off
end

