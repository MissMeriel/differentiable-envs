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
qualitycompare = readtable("/mnt/array/Home/Data/HPSTA/differentiable-envs/quality_compare_test_fix_batch", opts);


%% Clear temporary variables
clear opts
%%
minVal =-0.01;
qualitycompare.mw_ours_clamped = qualitycompare.mw_ours;
qualitycompare.rmw_ours_clamped = qualitycompare.rmw_ours;
qualitycompare.mw_ours_clamped(qualitycompare.mw_ours_clamped<minVal) = minVal;
qualitycompare.rmw_ours_clamped(qualitycompare.rmw_ours_clamped<minVal) = minVal ;

maxVal = 0.02;
qualitycompare.cf_ours_clamped = qualitycompare.cf_ours;
qualitycompare.rcf_ours_clamped = qualitycompare.rcf_ours;
qualitycompare.cf_ours_clamped(qualitycompare.cf_ours_clamped>maxVal) = maxVal;
qualitycompare.rcf_ours_clamped(qualitycompare.rcf_ours_clamped>maxVal) = maxVal ;

qualitycompare_orig = qualitycompare;
is_valid = qualitycompare_orig.status==0 & isfinite(qualitycompare_orig.mw_ours);
qualitycompare = qualitycompare_orig(is_valid,:);
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
figure; plotHeatScatter(qualitycompare.cf_dexnet/2,qualitycompare.cf_ours,'cf_dexnet','cf_ours')
figure; plotHeatScatter(qualitycompare.cf_dexnet/2,qualitycompare.rcf_ours,'cf_dexnet','rcf_ours')
figure; plotHeatScatter(qualitycompare.rcf_dexnet/2,qualitycompare.rcf_ours,'rcf_dexnet','rcf_ours')
figure; plotHeatScatter(qualitycompare.cf_dexnet/2,qualitycompare.rcf_dexnet/2,'cf_dexnet','rcf_dexnet')
figure; plotHeatScatter(qualitycompare.cf_ours,qualitycompare.mw_ours_clamped,'cf_ours','mw_ours')
figure; plotHeatScatter(qualitycompare.rcf_ours,qualitycompare.rmw_ours_clamped,'rcf_ours','rmw_ours')
figure; plotHeatScatter(qualitycompare.rcf_ours_clamped,qualitycompare.rmw_ours_clamped,'rcf_ours','rmw_ours')
figure; plotHeatScatter(qualitycompare.cf_ours,qualitycompare.rcf_ours,'cf_ours','rcf_ours')
figure; plotHeatScatter(qualitycompare.rcf_ours/2,qualitycompare.('gqcnn'),'rcf_ours','gqcnn')
plot([0,1],[0.004,0.004],'k-')
figure; plotHeatScatter(qualitycompare.rcf_dexnet,qualitycompare.('gqcnn'),'rcf_dexnet','gqcnn')
plot([0,1],[0.004,0.004],'k-')

%%
figure('units','pixels','Position',[0,0,260,200])
plotHeatScatter(qualitycompare.rcf_ours_clamped/0.002,qualitycompare.('gqcnn'),'RCF Oracle','GQCNN 2.1')
exportgraphics(gcf, sprintf(fullfile('score_figure', 'all_scores.png')))

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

base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/june-experiments-gpu-split-0-ll4ma-qc100-match-col-dist-softmin-scale-params-try-harder/Co';
% grasps = {'grasp0'};
grasps = {'grasp_28'};
% experiment_setup  = '';
post_path = '';
% experiments = { 'cf-DOWN-GQCNN-UP-coll-up','cf-DOWN-GQCNN-UP', ...
% 'GQCNN-UP-coll-up','GQCNN-UP', 'cf-DOWN-GQCNN-UP-l2-down',...
% 'cf-DOWN-GQCNN-UP-laplace-down','Random_Fuzz' , 'cf-DOWN-mw-UP','cf-DOWN-mw-UP-coll-up'};
% experiments={'cf-sig-UP-GQCNN-DOWN-coll-up', 'cf-sig-DOWN-GQCNN-UP-l2-down', 'cf-sig-UP-GQCNN-DOWN-l2-down', 'cf-sig-DOWN-GQCNN-UP-coll-up'}
% experiments={'cf-sig-DOWN-GQCNN-UP-coll-up',...
% 'cf-sig-DOWN-GQCNN-UP',...
% 'cf-sig-DOWN-GQCNN-UP-l2-down',...
% 'cf-sig-DOWN-GQCNN-UP-laplace-down',...
% 'cf-DOWN-mw-UP-coll-up',...
% 'cf-DOWN-mw-UP',...
% 'GQCNN-UP-coll-up',...
% 'GQCNN-UP',...
% };
experiments={...
    'cf-sig-UP-GQCNN-DOWN-coll-up',...
'cf-sig-UP-GQCNN-DOWN-l2-down',...
'cf-sig-UP-GQCNN-DOWN-laplace-down',...
'cf-sig-UP-GQCNN-DOWN',...
'cf-UP-mw-DOWN-coll-up',...
'cf-UP-mw-DOWN',...
'GQCNN-DOWN-coll-up',...
'GQCNN-DOWN'};
% experiments = {'cf-UP-GQCNN-DOWN-coll-up','cf-UP-GQCNN-DOWN',...
 % 'GQCNN-DOWN-coll-up','GQCNN-DOWN', 'cf-UP-GQCNN-DOWN-l2-down',...
 % 'cf-UP-GQCNN-DOWN-laplace-down', 'Random_Fuzz', 'cf-UP-mw-UP','cf-UP-mw-UP-coll-up'};


%colors = {'r+','m+','y+','k+'};
colors = {'r','m','y','k','w','g','b','r','m'};
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
% experiment_setup = 'test_cf-mw-scale-grad'

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
% %post_path = 'lr-0';
% % grasps = {'grasp0'};%,'grasp1','grasp2'};
% for exp_ind = 1:numel(experiments)
%     for grasp_ind = 1:numel(grasps)
%         full_path = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path);
%         plotlossvectors(full_path,{x_value,y_value,'self_collision'},scales(exp_ind,:),handle,colors{exp_ind})
%     end
% end

%%
% 


% experiment_setup = 'test_cf-mw-scale-grad';
% grasps = {'grasp0','grasp1','grasp2'};

figure;
handle = gcf();
y_value = 'rcf_ours';
x_value = 'gqcnn';
plotHeatScatter(qualitycompare.rcf_ours_clamped,qualitycompare.(x_value),y_value,x_value)

% base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/adv/adv-grasp/';
% 
%title({experiment_setup;experiments{2}},Interpreter="none")


scales = [0.002, 1, 1];


%post_path = 'lr-0';
% grasps = {'grasp0','grasp1','grasp2'};
hs = [];
for exp_ind = 1:numel(experiments)
    for grasp_ind = 1:numel(grasps)
        full_path = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path);
        hs(end+1) = plotlossvectors(full_path,{y_value,x_value,'self_collision'},scales,handle,colors{exp_ind});
    end
end
figure(handle)
legend(hs, experiments, 'location','best','interpreter','none')
ylim([0,maxVal])

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
