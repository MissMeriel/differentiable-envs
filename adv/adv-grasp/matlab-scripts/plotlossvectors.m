function plotlossvectors(base, names, scales, figure_handle, color)

if nargin < 1
    base = 'adv/adv-grasp/test_cf-mw-scale-grad-no-robust/cf-UP-GQCNN-DOWN-coll-up/grasp2/lr-0/';
end
if nargin < 2
    plot_pca = true;
end
files = {dir(fullfile(base,'*.mat')).name};
[~,order] = sort(cellfun(@(s)sscanf(s, 'it-%f-grasp.mat'),files));
files = files(order);

loss_mag = zeros(0,3);
grad_mag = zeros(0,3);
for file_idx = 1:numel(files)
    file = files{file_idx};
    grasp_struct = load(fullfile(base,file));
    if isfield(grasp_struct,'loss_mag')
        loss_mag = cat(1,loss_mag,grasp_struct.loss_mag(~grasp_struct.optim_status(:,3),:) .* sign(scales));
        grad_mag = cat(1,loss_mag,grasp_struct.grad_mag(~grasp_struct.optim_status(:,3),:));
        grad_mag = grad_mag(1:size(loss_mag,1),:);
    end
end


if nargin > 3
    figure(figure_handle)
    hold on
    h1 = plot(loss_mag(1,2) * abs(scales(2)),loss_mag(1,1) * abs(scales(1)),['-o',color]);
    h2 = plot(loss_mag(:,2) * abs(scales(2)),loss_mag(:,1) * abs(scales(1)),['-*',color]);
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

figure
hold on
plot(loss_mag(:,1))
plot(loss_mag(:,2))
xlim([0,500])
legend(names(1:2), 'Location','best',Interpreter='none')
title('loss mag')


figure
hold on
plot(grad_mag(:,1))
plot(grad_mag(:,2))
xlim([0,500])
legend(names(1:2), 'Location','best',Interpreter='none')
title('grad mag')
