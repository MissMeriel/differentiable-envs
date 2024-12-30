% bar = readObj('data/new_barclamp.obj')
base = '/mnt/array/Home/Data/HPSTA/differentiable-envs/dexnet-loop-flip-gqcnn-sign/4e301737d057917e25c70fb1df3f879b';
experiment_setup = '';
title(experiment_setup,Interpreter="none")
%experiments = {'cf-DOWN-mw-UP-coll-up','cf-UP-mw-DOWN-coll-up'};
experiments = {'cf-DOWN-GQCNN-UP-coll-up','cf-UP-GQCNN-DOWN-coll-up'};
iter_ids = {'it-25','it-50','it-75','it-100','it-125','it-150','it-175','it-200','it-225','it-250','it-275','it-300','it-325','it-350','it-375','it-400','it-425','it-450','it-475'};
%grasps = {'grasp0'};
%post_path = 'lr-0';
post_path = '';
grasps = {'grasp_92'};
Manualview1 = struct();
% grasp0
% cpos = [0.0151    0.4299    0.1000];
% caz = -181.3787;
% cel =  11.4361;
% cva =  7.8155;
% ctarg = [-0.0044,    0.0111,   -0.0518];
% Manualview1.xlim = [-0.0644    0.0556];
% Manualview1.ylim = [-0.0489    0.0711];
% Manualview1.zlim = [-0.1518    0.0482];

% grasp2
% cpos = [0.2915    1.0540    0.1166];
% caz = 163.8458;
% cel =  3.8579;
% cva =  2.6052*3;
% ctarg = [-0.0137    0.0119   -0.0304];
% Manualview1.xlim = [-0.0739    0.0464];
% Manualview1.ylim = [-0.0496    0.0734];
% Manualview1.zlim = [-0.1478    0.0870];

cpos = [0.2915    1.0540    0.1166];
caz = 163.8458;
cel =  4.05;
cva =  2.6052*3;
ctarg = [.0007    0.0095    0.0051];
Manualview1.xlim = [  -0.0394    0.0408];
Manualview1.ylim = [-0.0315    0.0505];
Manualview1.zlim = [-0.0731    0.0834];

Manualview1.CameraViewAngleMode = 'manual';
Manualview1.CameraUpVectorMode = 'manual';
Manualview1.CameraPositionMode = 'manual';
Manualview1.view=[caz,cel];
Manualview1.cameraTarget=ctarg;
Manualview1.Projection = 'orthographic';
Manualview1.CameraPosition = cpos;



Manualview1.PositionConstraint = 'innerposition';
%Manualview1.CameraUpVector = [0     0     1];
Manualview1.CameraViewAngle = cva/3;
figure;
for exp_ind = 1:numel(experiments)
    for grasp_ind = 1:numel(grasps)
        im = {};
        output_gif = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path,'matlab.gif');
        for iter_idx = 1:numel(iter_ids)
            iter_id = iter_ids{iter_idx};
            full_path = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path,strcat(iter_id,'.obj'));
            bar = readObj(full_path);
            %

            
            hold on;
            trisurf(bar.f.v,bar.v(:,1),bar.v(:,2),bar.v(:,3))
            alpha 0.5
            % triangleCenter = zeros(size(bar.f.v));
            % for i = 1:size(bar.f.v,1)
            %     triangle = bar.v(bar.f.v(i,:)',:);
            %     triangleCenter(i,:) = mean(triangle,1);
            %     dir1 = triangle(2,1:3)-triangle(1,1:3);
            %     dir1 = dir1/norm(dir1);
            %     dir2 = triangle(3,1:3)-triangle(1,1:3);
            %     dir2 = dir2/norm(dir2);
            %     % text(triangleCenter(i,1),triangleCenter(i,2),triangleCenter(i,3),num2str(i-1));
            %     % normal = cross(dir1,dir2)*0.05/2;
            %     % quiver3(triangleCenter(i,1),triangleCenter(i,2),triangleCenter(i,3),normal(1),normal(2),normal(3),'b','LineWidth',1,'AutoScale','off' ,'MaxHeadSize',1)
            % end
            xlabel('x')
            ylabel('y')
            zlabel('z')
            %pcshow(bar.v(:,1:3))
            mat_path = fullfile(base,experiment_setup,experiments{exp_ind},grasps{grasp_ind},post_path,strcat(iter_id,'-grasp.mat'));
            torch_struct = load(mat_path);
            
            for grasp_index = 1:size(torch_struct.endpoints3D,2)
                ray_o = squeeze(torch_struct.endpoints3D(:,grasp_index,:));
                ray_d = squeeze(torch_struct.contact_points(:,grasp_index,:)) - ray_o;
                quiver3(ray_o(:,1),ray_o(:,2),ray_o(:,3),...
                    ray_d(:,1),ray_d(:,2),ray_d(:,3),'c','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
                ray_o = squeeze(torch_struct.contact_points(:,grasp_index,:));
                ray_d = squeeze(torch_struct.contact_normals(:,grasp_index,:))/100;
                quiver3(ray_o(:,1),ray_o(:,2),ray_o(:,3),...
                    ray_d(:,1),ray_d(:,2),ray_d(:,3),'m','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
            end

            %
            %
            %
           
            grad_tensor = torch_struct.param_grad;
            grad_tensor = -grad_tensor ./ reshape(vecnorm(reshape(grad_tensor, [3,3*size(bar.v,1)]),2,2),[3,1,1])/100;
            hold on;
            quiver3(squeeze(bar.v(:,1)),squeeze(bar.v(:,2)),squeeze(bar.v(:,3)),squeeze(grad_tensor(1,:,1))',squeeze(grad_tensor(1,:,2))',squeeze(grad_tensor(1,:,3))','r','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
            quiver3(squeeze(bar.v(:,1)),squeeze(bar.v(:,2)),squeeze(bar.v(:,3)),squeeze(grad_tensor(2,:,1))',squeeze(grad_tensor(2,:,2))',squeeze(grad_tensor(2,:,3))','g','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
            quiver3(squeeze(bar.v(:,1)),squeeze(bar.v(:,2)),squeeze(bar.v(:,3)),squeeze(grad_tensor(3,:,1))',squeeze(grad_tensor(3,:,2))',squeeze(grad_tensor(3,:,3))','b','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
            text(-0.0044,    -0.0211,   -0.0518,iter_id)
            set(gca,Manualview1)
            drawnow
            gif_name = output_gif;
            %exportgraphics(gcf,gif_name,'Append',true);
            frame = getframe(gcf);
            im{iter_idx} = frame2im(frame);
            [A,map] = rgb2ind(im{iter_idx},256);
            if iter_idx == 1
                imwrite(A,map,gif_name,"gif",LoopCount=Inf, ...
                        DelayTime=0.5)
            else
                imwrite(A,map,gif_name,"gif",WriteMode="append", ...
                        DelayTime=0.5)
            end
            hold off;
            cla
        end
        
    end
end
%%
% adv_loss = load("adv_loss.mat").adv_loss/100;
% dist_loss_tri= load("dist_loss_tri.mat").dist_loss/10000/100;
% dist_loss_edg= load("dist_loss_edg.mat").dist_loss/10000/100;
% dist_loss_vtx= load("dist_loss_vtx.mat").dist_loss/10000/100;
% quiver3(bar.v(:,1),bar.v(:,2),bar.v(:,3),adv_loss(:,1),adv_loss(:,2),adv_loss(:,3),'k','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
% quiver3(bar.v(:,1),bar.v(:,2),bar.v(:,3),dist_loss_tri(:,1),dist_loss_tri(:,2),dist_loss_tri(:,3),'r','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
% quiver3(bar.v(:,1),bar.v(:,2),bar.v(:,3),dist_loss_edg(:,1),dist_loss_edg(:,2),dist_loss_edg(:,3),'g','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
% quiver3(bar.v(:,1),bar.v(:,2),bar.v(:,3),dist_loss_vtx(:,1),dist_loss_vtx(:,2),dist_loss_vtx(:,3),'b','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
% unc_struct = load("uncon.mat");
% for ind = 1:size(unc_vtx,2)
%     vert = unc_vtx(2,ind)+1;tta
%     vert_coords = [bar.v(vert,1),bar.v(vert,2),bar.v(vert,3)];
%     grad = [dist_loss_vtx(vert,1),dist_loss_vtx(vert,2),dist_loss_vtx(vert,3)];
%     end_of_arrow = vert_coords+ [dist_loss_vtx(vert,1),dist_loss_vtx(vert,2),dist_loss_vtx(vert,3)]+randn(1,3)/700;
%     if norm(grad) > 0
%         text(end_of_arrow(1),end_of_arrow(2),end_of_arrow(3),num2str(unc_vtx(1,ind)));
%     end
% end

%%
% title('dist normal')
% diststable = readtable('../../dists_normal.txt',Delimiter=' ',ReadVariableNames=false);
% % diststable.Properties.VariableNames = {'dist','ind0','ind1','not_spd','b0_1','b0_2','b1_1','b1_2','r','regularizer'};
%
% % title('dist')
% % diststable = readtable('../../dists.txt',Delimiter=' ',ReadVariableNames=false);
% diststable.Properties.VariableNames = {'dist','ind0','ind1','not_spd','b0_1','b0_2','b1_1','b1_2','regularizer'};
% diststable.constrained = diststable.b0_1 > -1/3 & diststable.b0_2 > -1/3 & ...
%     diststable.b1_1 > -1/3 & diststable.b1_2 > -1/3 & ...
%     diststable.b0_1 + diststable.b0_2 < 1/3 & ...
%     diststable.b1_1 + diststable.b1_2 < 1/3;
% %%
%
% % pcshow(bar.v,'markersize',50)
% % hold on
% %%
% %filterMask = [6088, 6089, 6090]
% %filterMask = find(diststable.ind0 == 366 & diststable.ind1 ==374,5);
% %filterMask = find(diststable.constrained,20);
% dists = diststable.dist;
% dists(~diststable.constrained) = inf;
% [~,filterMask] = mink(dists,5)
% % filterMask = [find(diststable.dist < (1 * 10^-6))]
%
% % symmetric
% %filterMask = [filterMask; filterMask+size(diststable,1)/2]
% hold on
% diststableFiltered = diststable(filterMask,:);
%
%
% % W vector between triangle corners 0
% % quiver3( ...
% %     bar.v(bar.f.v(diststableFiltered.ind0+1,1),1), ...
% %     bar.v(bar.f.v(diststableFiltered.ind0+1,1),2), ...
% %     bar.v(bar.f.v(diststableFiltered.ind0+1,1),3), ...
% %     bar.v(bar.f.v(diststableFiltered.ind1+1,1),1)-bar.v(bar.f.v(diststableFiltered.ind0+1,1),1), ...
% %     bar.v(bar.f.v(diststableFiltered.ind1+1,1),2)-bar.v(bar.f.v(diststableFiltered.ind0+1,1),2), ...
% %     bar.v(bar.f.v(diststableFiltered.ind1+1,1),3)-bar.v(bar.f.v(diststableFiltered.ind0+1,1),3), ...
% %     '--','AutoScale','off' ,'MaxHeadSize',1,'LineWidth',5)
% % % vector between triangle centers
% % quiver3( ...
% %     triangleCenter(diststableFiltered.ind0+1,1), ...
% %     triangleCenter(diststableFiltered.ind0+1,2), ...
% %     triangleCenter(diststableFiltered.ind0+1,3), ...
% %     triangleCenter(diststableFiltered.ind1+1,1)-triangleCenter(diststableFiltered.ind0+1,1), ...
% %     triangleCenter(diststableFiltered.ind1+1,2)-triangleCenter(diststableFiltered.ind0+1,2), ...
% %     triangleCenter(diststableFiltered.ind1+1,3)-triangleCenter(diststableFiltered.ind0+1,3), ...
% %     '-.','AutoScale','off' ,'MaxHeadSize',1,'LineWidth',5)
%
% b0 = [1/3 - diststableFiltered.b0_1 - diststableFiltered.b0_2, diststableFiltered.b0_1+1/3, diststableFiltered.b0_2+1/3];
% b1 = [1/3 - diststableFiltered.b1_1 - diststableFiltered.b1_2, diststableFiltered.b1_1+1/3, diststableFiltered.b1_2+1/3];
%
% dist_compare = array2table([diststableFiltered.dist, ...
%     vecnorm(bar.v(bar.f.v(diststableFiltered.ind1+1,1),:) - bar.v(bar.f.v(diststableFiltered.ind0+1,1),:),2,2).^2,...
%     vecnorm(triangleCenter(diststableFiltered.ind1+1,:)-triangleCenter(diststableFiltered.ind0+1,:),2,2).^2,...
%     zeros(size(diststableFiltered.dist))],'VariableNames',{'pytorch','corner0','center','pytorchbary'});
%
% % QP result vector
%
% for distindex = 1:size(diststableFiltered,1)
%     triangle0 = bar.v(bar.f.v(diststableFiltered.ind0(distindex)+1,:)',:);
%     point0 = b0(distindex,:) * triangle0;
%     triangle1 = bar.v(bar.f.v(diststableFiltered.ind1(distindex)+1,:)',:);
%     point1 = b1(distindex,:) * triangle1;
%     min_vec = point1-point0;
%     dist_compare.pytorchbary(distindex) = norm(min_vec)^2;
%     plot3(triangle0([1,2,3,1],1),...
%         triangle0([1,2,3,1],2),...
%         triangle0([1,2,3,1],3),'LineWidth',4)
%     plot3(triangle1([1,2,3,1],1),...
%         triangle1([1,2,3,1],2),...
%         triangle1([1,2,3,1],3),'--','LineWidth',4)
%
%     dir1 = triangle0(2,1:3)-triangle0(1,1:3);
%     %dir1 = dir1/norm(dir1);
%     dir2 = triangle0(3,1:3)-triangle0(1,1:3);
%     %dir2 = dir2/norm(dir2);
%     normal0 = cross(dir1,dir2);
%
%     % test constraint
%     eq_error = (point1 - point0) ./ normal0;
%     eq_error = eq_error/mean(eq_error);
%
%     if (abs(eq_error-1)>0.01)
%     quiver3( ...
%     point0(1), ...
%     point0(2), ...
%     point0(3), ...
%     min_vec(1), ...
%     min_vec(2), ...
%     min_vec(3), ...
%     '-.','AutoScale','off' ,'MaxHeadSize',1,'LineWidth',5)
%     else
%             quiver3( ...
%     point0(1), ...
%     point0(2), ...
%     point0(3), ...
%     min_vec(1), ...
%     min_vec(2), ...
%     min_vec(3), ...
%     '-','AutoScale','off' ,'MaxHeadSize',1,'LineWidth',5)
%     end
%
%     text( ...
%         (point0(1)+point1(1))/2, ...
%         (point0(2)+point1(2))/2, ...
%         (point0(3)+point1(3))/2, ...
%     strcat('from:',num2str(diststableFiltered.ind0(distindex)),':',num2str(diststableFiltered.dist(distindex))))
% end