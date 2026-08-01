function [loss_table, depth_image, depth_iter] = load_loss_folder(folder_path)
if nargin < 1
    folder_path = '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-1-kit/HerbSalt_800_tex/cf-UP-GQCNN-DOWN-coll-up/grasp_92';
end


files = {dir(fullfile(folder_path,'it-*.mat')).name};
[~,order] = sort(cellfun(@(s)sscanf(s, 'it-%f-grasp.mat'),files));
files = files(order);
if numel(files) < 2
    error('missing data in %s',folder_path)
end
loss_mag = zeros(0,0);
qual_measures_raw = zeros(0,0);
optim_status = zeros(0,0);
depth_image = zeros(0,32,32);
depth_iter = [];
for file_idx = 1:numel(files)
    file = files{file_idx};
    grasp_struct = load(fullfile(folder_path,file));
    if isfield(grasp_struct,'depth_image')
        depth_image(end+1,:,:) = squeeze(grasp_struct.depth_image);
        depth_iter(end+1) = sscanf(file, 'it-%f-grasp.mat');
    end
    if isfield(grasp_struct,'loss_mag')
        qual_measures_raw = cat(1,qual_measures_raw,grasp_struct.qual_measures_raw);
        optim_status = cat(1, optim_status, grasp_struct.optim_status);
        %optim_status = optim_status(1:size(qual_measures_raw,1),:);
    end
end

[~,ind] = unique(optim_status(:,1),'last');
if size(qual_measures_raw,2) > 8
loss_table = array2table(cat(2,qual_measures_raw,double(optim_status(ind,:))), 'VariableNames', ...
    {'gqcnn','rmw_ours','rcf_ours','self_collision','fingersDidHit','min_dist',...
    'l2_norm','laplacian_loss','step','mult','quad','alpha','iteration','steps_from_init','attempts','resets'});
else
loss_table = array2table(cat(2,qual_measures_raw,double(optim_status(ind,:))), 'VariableNames', ...
    {'gqcnn','rmw_ours','rcf_ours','self_collision','fingersDidHit','min_dist',...
    'l2_norm','laplacian_loss','iteration','steps_from_init','attempts','resets'});
end
%iter = 1:size(qual_measures_raw,1);



end