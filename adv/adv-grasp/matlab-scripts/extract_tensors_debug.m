directory = '/mnt/array/Home/Data/HPSTA/differentiable-envs/debug_throwaway/debug_mesh_attack';

%grasps = {dir(fullfile(directory,'*_batch.csv')).name};
grasps = {'94e289c89059106bd8f74b0004a598cd_grasp_90_batch.csv'};

for grasp = grasps
    grasp = grasp{1};
    disp(grasp)
    batch_path = fullfile(directory,grasp);
    path = fullfile(directory,erase(grasp,'_batch'));
    path_after = fullfile(directory,strrep(grasp,'_batch','_batch_after'));
    depth = csvread(path);
    depth_batch = csvread(batch_path);
    depth_batch_after = csvread(path_after);
    figure;
    tcl = tiledlayout(2,3);
    nexttile(1)
    imagesc(depth); colorbar;
    title('depth no batch')
    nexttile(2)
    imagesc(depth_batch); colorbar;
    title('depth batch (before)')
    nexttile(3)
    imagesc(depth_batch-depth); colorbar;
    title('depth batch diff (before)')
    nexttile(5)
    imagesc(depth_batch_after); colorbar;
    title('depth batch (after)')
    nexttile(6)
    imagesc(depth_batch_after-depth); colorbar;
    title('depth batch diff (after)')
    title(tcl,grasp,'interpreter','none')
end

%%
root_dir = '/mnt/array/Home/Data/HPSTA/differentiable-envs';
files = {'trans_then_rot_batch.csv',...
'rot_only_batch.csv',...
'trans_batch.csv',...
'resize_batch.csv',...
'cropped_batch.csv';
'trans_then_rot.csv',...
'rot_only_batch.csv',...
'trans.csv',...
'resize.csv',...
'cropped.csv'}';
figure
tl = tiledlayout(3,5);
title(tl,'translate align false')
for file_ind = 1:size(files,1)
    file = files{file_ind};
    depth = csvread(fullfile(root_dir,file));
    nexttile(file_ind);
    depth(depth==0) = nan;
    imagesc(depth,[0.57,0.62]); colorbar;
    axis image
    title(file,Interpreter="none")

    file_ind = file_ind + 5;
    file = files{file_ind};
    depth_batch = csvread(fullfile(root_dir,file));
    depth_batch(depth_batch==0) = nan;
    nexttile(file_ind);
    imagesc(depth_batch,[0.57,0.62]); colorbar;
    axis image
    title(file,Interpreter="none")

    file_ind = file_ind + 5;
    nexttile(file_ind);
    imagesc(depth_batch-depth); colorbar;
    axis image
    title(file,Interpreter="none")
end
