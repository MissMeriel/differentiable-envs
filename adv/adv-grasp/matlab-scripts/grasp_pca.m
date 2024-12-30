function grasp_pca(base, color, plot_pca)
%base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/dexnet-loop-500-iter-full-hull+robust/f6e6117261dca163713c042b393cc65b/oracle-grad-UP-minweight-DOWN/grasp_82';
%base = 'adv/adv-grasp/test_cf-mw-scale-grad/cf-UP-mw-DOWN-coll-up/grasp0/lr-0/'
if nargin < 1
    base = 'dexnet-loop-debug//7cde0fe08897826bc8635ea1c31dd83b/cf-DOWN-GQCNN-UP-coll-up/grasp_29/';
end
if nargin < 2
    plot_pca = true;
end
files = {dir(fullfile(base,'*.mat')).name};
%files = {'it-0-grasp.mat','it-250-grasp.mat','it-500-grasp.mat'};
grasp_idx = 1;
[~,order] = sort(cellfun(@(s)sscanf(s, 'it-%f-grasp.mat'),files));
files = files(order);
if plot_pca
    figure;
    tiledlayout(ceil(sqrt(numel(files))),ceil(sqrt(numel(files))))
end
handle = gcf();
compare_vec = zeros(numel(files),3);
for file_idx = 1:(numel(files))
    file = files{file_idx};
    grasp_struct = load(fullfile(base,file));
    batch = size(grasp_struct.grasp_matrix,3);
    grasp_mat_flat = permute(reshape(grasp_struct.grasp_matrix,[20,batch,6]),[2,1,3]);
    
    first_grasp = squeeze(grasp_mat_flat(grasp_idx,:,:));
    
    % if using matlab to compute hull
    % hull = convhulln(first_grasp,{'QJ'});

    % hull_unwrapped = reshape(first_grasp(hull(:),:),[size(hull,1), 6, 6]);
    % hull_unwrapped(:,end+1, :) = 1;
    % hull_unwrapped(:,end, end-2) = 1;
    % hull_unwrapped(:,:, end+1) = 1;
    % hull_unwrapped(:,end,end)=0;
    planes_cell = {};
    
    hull = grasp_struct.hull_simplices{grasp_idx}+1;
    hull_unwrapped = reshape(first_grasp(hull(:),:),[size(hull,1), 6, 6]);
    hull_unwrapped(:,end+1, :) = 1;
    hull_unwrapped(:,end, end-2) = 1;
    hull_unwrapped(:,:, end+1) = 1;
    hull_unwrapped(:,end,end)=0;
    area = zeros(size(hull_unwrapped,1),1);
    for i = 1:size(hull_unwrapped,1)
        facet = reshape(hull_unwrapped(i,:,:),[7,7]);
        area(i) = abs(det(facet(1:6,1:6)));
        det_val = 1;
        % if using matlab for convhull
        % planes_cell{i} = facet \ [0;0;0;0;0;0;det_val] ;
        % planes_cell{i} = planes_cell{i} / vecnorm(planes_cell{i}(1:6),2,1);
    end

    
    % if using matlab convex hull
    % planes = cat(2,planes_cell{:});
    planes = grasp_struct.hull_equations{grasp_idx}';
    % if only comfortable with normal, not offset, can recompute offset
    % dist = sum(planes(1:6,:)' .* squeeze(hull_unwrapped(:,1,1:6)),2);
    dist = planes(7,:)';
    A = first_grasp;
    A(end+1,:) = 0;
    A(:,end+1) = 1;
    b = zeros(size(first_grasp,2),1);
    b(end+1,1) = 1;
    G = -eye(size(first_grasp,1));
    G(:,end+1) = 1;
    h = zeros(size(first_grasp,1),1);
    f = zeros(size(first_grasp,1),1);
    f(end+1,1) = -1;
    alpha = linprog(f,G,h,A',b');
    constrained = alpha(1:(end-1)) == alpha(end);
    if file_idx == 1
        [coefs, ~,~,~,~,mu] =  pca(first_grasp,Centered=false);
    end
    cone_ind = [1:8,10:18];
    torsion_ind = [9:10,19:20];

    reduced = (first_grasp-mu) * coefs(:,1:2);

    dist_filt=dist;
    dist_filt(area < 0.0000001) = 100000;
    [cf,mindex] = min(abs(dist_filt));
    vertices_in_min = hull(abs(dist)==cf,:);
    active_plane = reduced(unique(vertices_in_min(:)),:);

    compare_vec(file_idx,1) = grasp_struct.quality(grasp_idx);
    compare_vec(file_idx,2) = alpha(end);
    compare_vec(file_idx,3) = cf;
    if plot_pca
        nexttile
        hold on
        plot(reduced(torsion_ind,1),reduced(torsion_ind,2), 'bo')
        plot(reduced(cone_ind,1),reduced(cone_ind,2), 'go')
        plot(reduced(constrained,1),reduced(constrained,2), 'xr')
        plot([0,0,0,0,0,0]*coefs(:,1),[0,0,0,0,0,0]*coefs(:,2),'k+')
        for ind = cone_ind
            text(reduced(ind,1)+0.5,reduced(ind,2), num2str(alpha(ind)))
        end
        for ind = torsion_ind
            text(reduced(ind,1)+0.5,reduced(ind,2), num2str(alpha(ind)))
        end
        plot(active_plane(:,1),active_plane(:,2),'+k')
        resultsstring = sprintf('stored: %f mw: %f cf: %f',grasp_struct.quality(grasp_idx), alpha(end), cf);
        title({ file;resultsstring})
    end
end
%%

figure; hold on
plot(1:numel(files),compare_vec(:,1)/compare_vec(1,1),'*-')
plot(1:numel(files),compare_vec(:,2)/compare_vec(1,2),'*-')
plot(1:numel(files),compare_vec(:,3)/compare_vec(1,3),'*-')
legend({'rcf-stored','mw','rcf'})
%%

% base = 'grasp_29';
% files = {dir(fullfile(base,'*.mat')).name};
% grasp_idx = 1;
% [~,order] = sort(cellfun(@(s)sscanf(s, 'it-%f-grasp.mat'),files));
% files = files(order);
% cone_ind = [1:8,10:18];
% torsion_ind = [9:10,19:20];
% ind_try = [cone_ind(floor(linspace(1,16,6))),torsion_ind(1)];
% %ind_try = [cone_ind(floor(linspace(1,16,5))),torsion_ind(1:2)];
% %ind_try = [cone_ind(floor(linspace(1,8,5))),torsion_ind(1:2)];
% %ind_try = [cone_ind(floor(linspace(1,16,5))),torsion_ind(floor(linspace(1,4,2)))];
% %ind_try = cone_ind(floor(linspace(1,16,7)));
% % figure;
% % tiledlayout(ceil(sqrt(numel(files))),ceil(sqrt(numel(files))))
% volume = zeros(numel(files),20);
% for file_idx = 1:(numel(files))
% for grasp_idx = 1:20
% file = files{file_idx};
% grasp_struct = load(fullfile(base,file));
% grasp_mat_flat = permute(reshape(grasp_struct.grasp_matrix,[20,25,6]),[2,1,3]);
% simplex = squeeze(grasp_mat_flat(grasp_idx, ind_try, :));
% simplex(:,end+1) = 1;
% volume(file_idx,grasp_idx) = det(simplex);
% end
% 
% end
% figure
% histogram(volume(:))
% title(ind_try)