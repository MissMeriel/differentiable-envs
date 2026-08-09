experiment_setup  = '';

% bases = {'/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-0-3dnet',...
%     '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-1-kit',...
%     '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-0-kit'};
bases = {'/mnt/array/Home/Data/HPSTA/differentiable-envs/Sep-experiments-gpu-split-0-mine-play-with-constraint-prune-vert-0'}
% grasps = {'grasp0'};

% experiment_setup  = '';
post_path = '';

% experiments = {'cf-UP-GQCNN-DOWN-coll-up','cf-UP-GQCNN-DOWN', 'GQCNN-DOWN-coll-up','GQCNN-DOWN',...
%     'cf-UP-GQCNN-DOWN-l2-down', 'cf-UP-GQCNN-DOWN-laplace-down',...
%     'Random_Fuzz','cf-UP-mw-UP','cf-UP-mw-UP-coll-up',...
%     'cf-DOWN-GQCNN-UP-coll-up','cf-DOWN-GQCNN-UP', 'GQCNN-UP-coll-up','GQCNN-UP',...
%     'cf-DOWN-GQCNN-UP-l2-down', 'cf-DOWN-GQCNN-UP-laplace-down',...
%     'Random_Fuzz','cf-DOWN-mw-UP','cf-DOWN-mw-UP-coll-up'};
experiments = {};
for base_id = 1:numel(bases) 
        base = bases{base_id};

    objects = {dir(base).name};
    filter_files = {'.','..','debug_mesh_attack'};
    filter = true(size(objects));
    for filter_file = filter_files
        filter = filter & ~ strcmp(objects, filter_file); 
    end
    objects = objects(filter);
   for object = objects
               object_dir = fullfile(base,object{1});
        
            % check completion
        experiments_local = dir(object_dir);
        experiments_local=experiments_local(~ismember({experiments_local.name},{'.','..'}));
        experiments_local = {experiments_local.name};
        experiments = union(experiments,experiments_local);
   end
end
%signs = [-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1];
signs = (double(contains(experiments, 'GQCNN-UP') | contains(experiments, 'cf-DOWN'))-0.5)*2;
best_rows = {};
best_no_collide_rows = {};
first_rows = {};
for exper_id = 1:numel(experiments)
    best_rows{exper_id} = table();
    first_rows{exper_id} = table();
    best_no_collide_rows{exper_id} = table();
end

for base_id = 1:numel(bases)
    base = bases{base_id};

    objects = {dir(base).name};
    filter_files = {'.','..','debug_mesh_attack'};
    filter = true(size(objects));
    for filter_file = filter_files
        filter = filter & ~ strcmp(objects, filter_file); 
    end
    objects = objects(filter);

    for object = objects
        object_dir = fullfile(base,object{1});
        
        

        grasps_complete = {};
        for experiment = experiments'
            experiment_dir = fullfile(object_dir, experiment{1});
            grasps = {dir(fullfile(experiment_dir,'grasp_*')).name};
    
            for grasp = grasps
                grasp_dir = fullfile(experiment_dir,grasp{1});
                try 
                    losses = load_loss_folder(grasp_dir);
                catch
                    continue
                end
                grasps_complete{end+1} = grasp{1};
            end
        end
        if numel(grasps_complete) > 0
            [d, id] = findgroups(grasps_complete);
            counts = histcounts(d);
            grasps = id(counts == numel(experiments));
            if numel(grasps)>0
                disp(object{1})
            end
            for exper_id = 1:numel(experiments)
                for grasp_id = 1:numel(grasps)
                    grasp = grasps{grasp_id};
                    experiment_dir = fullfile(object_dir, experiments{exper_id});
                    grasp_dir = fullfile(experiment_dir,grasp);
                    losses = load_loss_folder(grasp_dir);
                    losses.self_collision = double(losses.self_collision);
                    losses.('object') = repmat(object(1),size(losses,1),1);
                    losses.('grasp') = repmat({grasp},size(losses,1),1);
                    losses.('grasp_dir') = repmat({grasp_dir},size(losses,1),1);
                    losses.("diff") = signs(exper_id)*(losses.gqcnn - min(losses.rcf_ours,2));
                    losses.("count_col") = repmat(sum(losses.min_dist <= 1e-10),size(losses,1),1)/size(losses,1);
                    [~,best_idx] = max(losses.("diff"));
    
                    best_rows{exper_id} = cat(1, best_rows{exper_id}, losses(best_idx,:));
                    first_rows{exper_id} = cat(1, first_rows{exper_id}, losses(1,:));
                    losses_no_collide = losses(losses.min_dist > 1e-10,:);
                    [~,best_idx] = max(losses_no_collide.("diff"));
                    best_no_collide_rows{exper_id} = cat(1, best_no_collide_rows{exper_id}, losses_no_collide(best_idx,:));
    
                end
            end
        end
    end
end
%%
[unique_vals,unique_inds] = unique(first_rows{1}.object);
fprintf('%i grasps,   %i objs\n',size(first_rows{1},1), numel(unique_vals))

first_rows_unique = {};
best_rows_unique = {};
best_no_collide_rows_unique = {};
for i = numel(first_rows):-1:1
first_rows_unique{i} = first_rows{i}(unique_inds,:);
best_rows_unique{i} = best_rows{i}(unique_inds,:);
best_no_collide_rows_unique{i} = best_no_collide_rows{i}(unique_inds,:);
end

[unique_vals,unique_inds] = unique(first_rows_unique{1}.object);
fprintf('%i grasps,   %i objs after filter\n',size(first_rows_unique{1},1), numel(unique_vals))

fprintf('Avg + diff (coll), Avg L2, Ratio Coll,  Avg + diff,   Avg L2 \n')
for exper_id = 1:numel(experiments)
    diff_diff = best_rows_unique{exper_id}.diff - first_rows_unique{exper_id}.diff;
    diff_diff(diff_diff > 1) = 1;
    diff_diff_no_coll = best_no_collide_rows_unique{exper_id}.diff - first_rows_unique{exper_id}.diff;
    diff_diff_no_coll(diff_diff_no_coll > 1) = 1;
    nd = fitdist(diff_diff,'Normal');
    nd_ci = nd.paramci;
    nd_no = fitdist(diff_diff_no_coll,'Normal');
    nd_no_ci= nd_no.paramci;
    fprintf('%0.3f +- %0.3f,    %0.3f,     %0.3f,   %0.3f +- %0.3f, %0.3f,  %s \n',...
        nd.mu, nd.mu-nd_ci(1,1), mean(best_rows_unique{exper_id}.l2_norm),...
        mean(best_rows_unique{exper_id}.count_col),...
        nd_no.mu,nd_no.mu-nd_no_ci(1,1), mean(best_no_collide_rows_unique{exper_id}.l2_norm),...
        experiments{exper_id})
end
%%
figure('units','pixels','Position',[0,0,220,200])
maxVal = 0.02;
rcf_clamped = first_rows_unique{1,1}.rcf_ours*0.002;
rcf_clamped(rcf_clamped > maxVal) = maxVal;

plot(first_rows_unique{1,1}.gqcnn, rcf_clamped/0.002,'.k')
ylabel('RCF Oracle','Interpreter','latex')
xlabel('GQCNN 2.1','Interpreter','latex')
set(gca,"TickLabelInterpreter",'latex')
exportgraphics(gcf, sprintf(fullfile('score_figure', 'sample_starting_scores.png')))

%%
coll_discrepency = {};
for i = 1:14
coll_discrepency{i} = best_rows_unique{1,i};
coll_discrepency{i}.discrepency = best_rows_unique{1,i}.diff - best_no_collide_rows_unique{1,i}.diff;
end

oracle_discrepency = {};
for i = 1:14
    oracle_discrepency{i} = best_rows_unique{1,i};
    if i >= 8
        j = 8;
        oracle_discrepency{i}.rcf_diff = best_no_collide_rows_unique{1,i}.rcf_ours - first_rows_unique{1,j}.rcf_ours;

    else
        j = 1;
        oracle_discrepency{i}.rcf_diff = first_rows_unique{1,j}.rcf_ours - best_no_collide_rows_unique{1,i}.rcf_ours;

    end

oracle_discrepency{i}.discrepency = best_no_collide_rows_unique{1,j}.diff - best_no_collide_rows_unique{1,i}.diff;
end