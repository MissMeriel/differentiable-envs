%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 12);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["object", "grasp", "status", "cf_dexnet", "cf_ours", "rcf_dexnet", "rcf_ours", "mw_ours", "rmw_ours", "gqcnn", "in_collision", "is_watertight", "is_inverted"];
opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double","categorical", "categorical", "categorical"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["object", "is_watertight", "is_inverted"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "grasp", "TrimNonNumeric", true);
opts = setvaropts(opts, "grasp", "ThousandsSeparator", ",");

% Import the data
qualitycompare = readtable("/mnt/array/Home/Data/HPSTA/differentiable-envs/quality_compare_test_gqcnn_swap.csv", opts);


%% Clear temporary variables
clear opts
%%
minVal =-0.01;
qualitycompare.mw_ours_clamped = qualitycompare.mw_ours;
qualitycompare.rmw_ours_clamped = qualitycompare.rmw_ours;
qualitycompare.mw_ours_clamped(qualitycompare.mw_ours_clamped<minVal) = minVal;
qualitycompare.rmw_ours_clamped(qualitycompare.rmw_ours_clamped<minVal) = minVal ;
qualitycompare_orig = qualitycompare;
qualitycompare = qualitycompare_orig(qualitycompare_orig.status==0 & isfinite(qualitycompare_orig.mw_ours),:);
%% Clear temporary variables
clearvars filename formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp rawNumericColumns rawStringColumns R catIdx idx;

%%
% figure
% hold on
% plot(qualitycompare.cf_dexnet,qualitycompare.rcf_dexnet,'b.')
% plot(qualitycompare.cf_dexnet, qualitycompare.cf_ours,'r.')
% 
% xlabel('dexnet canny ferrari')
% legend({'our cf','dexnet rcf'},'Location','best')
%%
figure; plotHeatScatter(qualitycompare.cf_dexnet*2,qualitycompare.cf_ours,'cf_dexnet','cf_ours')
figure; plotHeatScatter(qualitycompare.cf_dexnet*2,qualitycompare.rcf_ours,'cf_dexnet','rcf_ours')
figure; plotHeatScatter(qualitycompare.rcf_dexnet*2,qualitycompare.rcf_ours,'rcf_dexnet','rcf_ours')
figure; plotHeatScatter(qualitycompare.cf_dexnet*2,qualitycompare.rcf_dexnet*2,'cf_dexnet','rcf_dexnet')
figure; plotHeatScatter(qualitycompare.cf_ours,qualitycompare.rcf_ours,'cf_ours','rcf_ours')
figure; plotHeatScatter(qualitycompare.rcf_ours/2,qualitycompare.('gqcnn'),'rcf_ours','gqcnn')
plot([0,1],[0.004,0.004],'k-')
figure; plotHeatScatter(qualitycompare.rcf_dexnet,qualitycompare.('gqcnn'),'rcf_dexnet','gqcnn')
plot([0,1],[0.004,0.004],'k-')

%%
%
% base ='dexnet-loop-debug//7cde0fe08897826bc8635ea1c31dd83b/';
% grasps = {'grasp_29'};
% experiment_setup  = '';
% post_path = '';

% grasps = {'grasp0'};%,'grasp1','grasp2'};
% post_path = 'lr-0';
% base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/adv/adv-grasp/auto-grad-scale';

experiment_setup  = '';

% base = 'dexnet-loop-lower-collision-saturate-higher-lr//sqbowl/';
base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/dexnet-loop-flip-gqcnn-sign/4e301737d057917e25c70fb1df3f879b';
% grasps = {'grasp_77'};
grasps = {'grasp_92'};
% experiment_setup  = '';
post_path = '';
%% robust
%experiment_setup = 'test_cf-mw-scale-grad-no-robust';
% experiments = {'cf-UP-mw-DOWN-coll-up', 'cf-DOWN-mw-UP-coll-up'}
% 
% figure;
% handle = gcf();
% x_value = 'cf_ours';
% y_value = 'mw_ours';
% plotHeatScatter(qualitycompare.(x_value),qualitycompare.mw_ours_clamped,x_value,y_value)
% title(experiment_setup,Interpreter="none")
% scales = [[-0.002, 0.002, -1];[0.002, -0.002, -1]];
% colors = {'r','m'};
% 
% for exp_ind = 1:numel(experiments)
%     for grasp_ind = 1:numel(grasps)
%         full_path = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path);
%         plotlossvectors(full_path,{x_value,y_value,'self_collision'},scales(exp_ind,:),handle,colors{exp_ind})
%     end
% end
% 
% %%
% % experiment_setup = 'test_cf-mw-scale-grad'
% 
% figure;
% handle = gcf();
% x_value = 'rcf_ours';
% y_value = 'rmw_ours';
% plotHeatScatter(qualitycompare.(x_value),qualitycompare.rmw_ours_clamped,x_value,y_value)
% % base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/adv/adv-grasp';
% ;scales = [[-0.002, 0.002, -1];[0.002, -0.002, -1]];
% % ;
% title(experiment_setup,Interpreter="none")
% colors = {'r','m'};
% % post_path = 'lr-0';
% % grasps = {'grasp0'};%,'grasp1','grasp2'};
% for exp_ind = 1:numel(experiments)
%     for grasp_ind = 1:numel(grasps)
%         full_path = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path);
%         plotlossvectors(full_path,{x_value,y_value,'self_collision'},scales(exp_ind,:),handle,colors{exp_ind})
%     end
% end

%%

experiments = {'cf-UP-GQCNN-DOWN-coll-up', 'cf-DOWN-GQCNN-UP-coll-up'};
% experiment_setup = 'test_cf-mw-scale-grad';
% grasps = {'grasp0','grasp1','grasp2'};

figure;
handle = gcf();
x_value = 'rcf_ours';
y_value = 'gqcnn';
plotHeatScatter(qualitycompare.(x_value),qualitycompare.(y_value),x_value,y_value)

% base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/adv/adv-grasp/';
% 
title(experiment_setup,Interpreter="none")


scales = [[-0.002, 1, -1];[0.002, -1, -1]];
colors = {'r','m'};
% post_path = 'lr-0';
% grasps = {'grasp0','grasp1','grasp2'};
for exp_ind = 1:numel(experiments)
    for grasp_ind = 1:numel(grasps)
        full_path = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path);
        plotlossvectors(full_path,{x_value,y_value,'self_collision'},scales(exp_ind,:),handle,colors{exp_ind})
    end
end

%%
% experiment_setup = 'test_cf-mw-scale-grad-no-robust';
% 
% grasps = {'grasp2'};
% 
% figure;
% handle = gcf();
% x_value = 'cf_ours';
% y_value = 'gqcnn';
% plotHeatScatter(qualitycompare.(x_value),qualitycompare.(y_value),x_value,y_value)
% 
% title(experiment_setup,Interpreter="none")
% scales = [[0.002, -1, -1];[-0.002, 1, -1]];
% colors = {'r','m'};
% 
% post_path = 'lr-0';
% for exp_ind = 1:numel(experiments)
%     for grasp_ind = 1:numel(grasps)
%         full_path = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path);
%         plotlossvectors(full_path,{x_value,y_value,'self_collision'},scales(exp_ind,:),handle,colors{exp_ind})
%     end
% end
%%
% plotlossvectors('/mnt/array/Home/Data/HPSTA/differentiable-envs/adv/adv-grasp/test_cf-mw-scale-grad/cf-DOWN-GQCNN-UP-coll-up/grasp0/lr-0', ...
%     {'cf_ours','gqcnn','self_collision'},[[0.002, -1, -1]])
% title('Collision Gradient Projection and Loss')
% %%
% plotlossvectors('/mnt/array/Home/Data/HPSTA/differentiable-envs/adv/adv-grasp/test_cf-mw-scale-grad-no-collision-weight/cf-DOWN-GQCNN-UP-coll-up/grasp0/lr-0', ...
%     {'cf_ours','gqcnn','self_collision'},[[0.002, -1, -1]])
% title('Collision Gradient Projection Only')
% %%
% plotlossvectors('/mnt/array/Home/Data/HPSTA/differentiable-envs/adv/adv-grasp/test_cf-mw-scale-grad-no-collision-projection-0-scale/cf-DOWN-GQCNN-UP-coll-up/grasp0/lr-0', ...
%     {'cf_ours','gqcnn','self_collision'},[[0.002, -1, -1]])
% title('Collision Gradient Ignored')
%%
