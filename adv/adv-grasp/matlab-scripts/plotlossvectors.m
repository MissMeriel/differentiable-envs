function [h1, h2] = plotlossvectors(base, names, scales, figure_handle, color)

if nargin < 1
    % shoe
    %base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-0-3dnet/Co/cf-UP-GQCNN-DOWN-coll-up/grasp_14';
    % gqcnn starts at middle and rcf starts a little low (1.6)
    % base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-1-kit/OrangeMarmelade_800_tex/GQCNN-UP-coll-up/grasp_58';
    % hering tin
    % base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-1-kit/HeringTin_800_tex/cf-DOWN-GQCNN-UP-coll-up/grasp_87';
    % heel
    % base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-0-3dnet/9a17ca5037beb643e7e684d25d4dcaf01/cf-UP-GQCNN-DOWN-coll-up/grasp_20';
    base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-1-kit/ToyCarYelloq_800_tex/cf-UP-GQCNN-DOWN-coll-up/grasp_52';
    save_latex = 1;
    dire = 'loss_figure';
else
    save_latex = 0;
end
if nargin < 2
    plot_pca = true;
end
if nargin < 3
    scales = 1;
end

[losses, depth, depth_ind] = load_loss_folder(base);
if nargin > 3
    figure(figure_handle)
    hold on
    h1 = plot(losses{:,names{2}} * abs(scales(2)),losses{:,names{1}} * abs(scales(1)),['-',color]);
    h2 = plot(losses{1,names{2}} * abs(scales(2)),losses{1,names{1}} * abs(scales(1)),['-o',color(1)],'MarkerSize',12);
else
    h1 = 0;
    h2 = 0;
end

% figure
% hold on
% yyaxis left
% plot(loss_mag(:,1))
% plot(loss_mag(:,2))
% yyaxis right
% plot(loss_mag(:,3))
% xlim([0,500])
% legend(names, 'Location','best',Interpreter='none')

% figure
% hold on
% plot(iter,loss_mag(iter,1),color)
% plot(iter,loss_mag(iter,2),[color,'--'])
% plot(index,loss_mag(index,1),[color,'-*'])
% plot(index,loss_mag(index,2),[color,'--*'])
% xlim([0,500])
% legend(names(1:2), 'Location','best',Interpreter='none')
% title('loss mag')

exp_names = {};
for i = 1:3
    [base,exp_names{i}] = fileparts(base);
end

names_all = losses.Properties.VariableNames;

% figure
% hold on
% for name = names_all
%     if strcmp(name, 'self_collision')
%         yyaxis right
%     else
%         yyaxis left
%     end
% plot(iter,losses{:,name})
% end
% xlim([0,250])
% legend(names_all, 'Location','best',Interpreter='none')
% title('Raw')
min_diff = 100;
max_diff = -100;
inds = unique(floor(linspace(1,size(depth,1),5)));
inds = inds(2:end);
for ind = inds
    d_im = squeeze(depth(ind,:,:));
    d_im = d_im - squeeze(depth(1,:,:));
    min_diff = min([min(d_im(:)),min_diff]);
    max_diff = max([max(d_im(:)),max_diff]);
end

fh = figure('units','normalized','Position',[0,0,1,0.5]);
% figure
tl = tiledlayout(3,4,'TileSpacing','tight');
for ind = [inds,1]


    d_im = squeeze(depth(ind,:,:));
    if ind ~=1
        ax1 = nexttile();
        d_im = d_im - squeeze(depth(1,:,:));

        imagesc(d_im, [min_diff, max_diff]);
        colormap(ax1, spring)
        colorbar(ax1,'TickLabelInterpreter','latex')
    else
        ax2 = nexttile();

        imagesc(d_im)
        colormap(ax2, summer)
        colorbar(ax2,'TickLabelInterpreter','latex')

    end
    set(gca,"TickLabelInterpreter",'latex')
    axis image

    if ind == 1
        title({'Initial Depth Image',sprintf('iter %i gqcnn %0.2f rcf %0.2f coll: %i', depth_ind(ind), ...
            losses.gqcnn( depth_ind(ind)+1),losses.rcf_ours( depth_ind(ind)+1)) },'interpreter','latex')
    else
        title({'$\Delta$ from Initial Depth Image',sprintf('iter %i gqcnn %0.2f rcf %0.2f', depth_ind(ind), ...
            losses.gqcnn( depth_ind(ind)+1),losses.rcf_ours( depth_ind(ind)+1) )},'interpreter','latex')
    end
    if save_latex
        figure
        if ind ~=1
            ax1 = nexttile();

            imagesc(d_im, [min_diff, max_diff]);
            colormap(ax1, spring)
            if ind == 21
                colorbar(ax1,'TickLabelInterpreter','latex')
            end
        else
            ax2 = nexttile();

            imagesc(d_im)
            colormap(ax2, summer)
            colorbar(ax2,'TickLabelInterpreter','latex')
            

        end
        axis image
        set(gca,"TickLabelInterpreter",'latex')
        exportgraphics(gcf, sprintf(fullfile(dire, 'depth_%i.png'),ind))
    end

    figure(fh)
end
nexttile([1,3])
handles = plot_losses_impl(losses);
title(tl, exp_names,Interpreter='none')
nexttile(9)
semilogy(losses.iteration,losses.alpha)
hold on
xline(losses.iteration(losses.alpha==0))
title('alpha')
xlim([min(losses.iteration)-0.2,max(losses.iteration+0.2)])
nexttile(10)
plot(losses.iteration,losses.step)
title('step')
nexttile(11)
plot(losses.iteration,losses.mult)
title('multiplier')
nexttile(12)
plot(losses.iteration,losses.quad)
title('quadcoef')
if save_latex
    figure('units','pixels','Position',[0,0,400,200])
    plot_losses_impl(losses)
    exportgraphics(gcf, sprintf(fullfile(dire, 'plot.png'),ind))
end




    function handles = plot_losses_impl(losses)
        handles  = []; name_scale = {};
        hold on
        for name = {'min_dist','gqcnn','rcf_ours','rmw_ours'}
            if strcmp('gqcnn',name) | strcmp('rcf_ours',name) | strcmp('rmw_ours',name)
                loss_scaled = losses{:,name};%/losses{1,name};
                if strcmp('gqcnn',name)
                    yyaxis right
                else
                    yyaxis left
                end
                handles(end+1) = plot(losses.iteration,loss_scaled);
                name_display = strrep(name,'rcf_ours','RCF Oracle');
                name_display = strrep(name_display,'gqcnn','GQCNN 2.1');
                name_scale{end+1} = [name_display{1}, ' (' , num2str(losses{1,name}), ')'];


            elseif strcmp('min_dist',name)
                collisions = find(losses{:,name} < 1e-14)-1;
                if isempty(collisions)

                else
                    array_of_lines = xline(collisions);
                    handles(end+1) = array_of_lines(1);
                    name_scale{end+1} = 'Self-Collision Iters';
                end
                
            else

            end
        end
        % xlim([0,250])
        legend(handles, name_scale, 'Location','NorthEast',Interpreter='latex')
        xlabel('Iteration','Interpreter','latex')
        ylabel('Score','Interpreter','latex')
        set(gca,"TickLabelInterpreter",'latex')
    end
end
%title(tl, exp_names,Interpreter='none')
