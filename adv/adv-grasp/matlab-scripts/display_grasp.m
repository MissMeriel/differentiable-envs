function display_grasp(path, iter_id)

if nargin < 2
    iter_id = 'it-0';
end
if nargin < 1
    path = '/mnt/array/Home/Data/HPSTA/differentiable-envs/feb-experiments-gpu-split-0-3dnet/Co/cf-UP-GQCNN-DOWN-coll-up/grasp_14';
end
full_path = fullfile(path,strcat(iter_id,'.obj'));
bar = readObj(full_path);

hold on;
trisurf(bar.f.v,bar.v(:,1),bar.v(:,2),bar.v(:,3))
alpha 0.5
xlabel('x')
ylabel('y')
zlabel('z')
mat_path = fullfile(path,strcat(iter_id,'-grasp.mat'));
torch_struct = load(mat_path);

for grasp_index = 1:size(torch_struct.endpoints3D,2)
    ray_o = squeeze(torch_struct.endpoints3D(:,grasp_index,:));
    ray_d = squeeze(torch_struct.contact_points(:,grasp_index,:)) - ray_o;
    if grasp_index == 1
    fh_finger = quiver3(ray_o(:,1),ray_o(:,2),ray_o(:,3),...
        ray_d(:,1),ray_d(:,2),ray_d(:,3),'c','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
    fh_finger = fh_finger(1);
    else
        quiver3(ray_o(:,1),ray_o(:,2),ray_o(:,3),...
        ray_d(:,1),ray_d(:,2),ray_d(:,3),'c','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
    end
    ray_o = squeeze(torch_struct.contact_points(:,grasp_index,:));
    ray_d = squeeze(torch_struct.contact_normals(:,grasp_index,:))/100;
    if grasp_index == 1
    fh_normal = quiver3(ray_o(:,1),ray_o(:,2),ray_o(:,3),...
        ray_d(:,1),ray_d(:,2),ray_d(:,3),'m','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
    fh_finger = fh_finger(1);
    else
        quiver3(ray_o(:,1),ray_o(:,2),ray_o(:,3),...
        ray_d(:,1),ray_d(:,2),ray_d(:,3),'m','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
    end
end

if isfield(torch_struct, 'param_grad') & ~strcmp(iter_id, 'it-0')

    grad_tensor = torch_struct.param_grad;
    num_objectives = size(grad_tensor,1);
    grad_tensor = -grad_tensor ./ reshape(vecnorm(reshape(grad_tensor, [num_objectives,3*size(bar.v,1)]),2,2),[num_objectives,1,1])/100;
    hold on;
    quiver3(squeeze(bar.v(:,1)),squeeze(bar.v(:,2)),squeeze(bar.v(:,3)),squeeze(grad_tensor(1,:,1))',squeeze(grad_tensor(1,:,2))',squeeze(grad_tensor(1,:,3))','r','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
    if num_objectives > 1
        quiver3(squeeze(bar.v(:,1)),squeeze(bar.v(:,2)),squeeze(bar.v(:,3)),squeeze(grad_tensor(2,:,1))',squeeze(grad_tensor(2,:,2))',squeeze(grad_tensor(2,:,3))','g','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
        if num_objectives > 2
             quiver3(squeeze(bar.v(:,1)),squeeze(bar.v(:,2)),squeeze(bar.v(:,3)),squeeze(grad_tensor(3,:,1))',squeeze(grad_tensor(3,:,2))',squeeze(grad_tensor(3,:,3))','b','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
        end
    end


end

    if contains(path,'coll-up') && contains(path,'cf-')
        
    fh1 = quiver3(0,0,0,0,0,0,'r','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
    fh2 = quiver3(0,0,0,0,0,0,'g','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
    fh3 = quiver3(0,0,0,0,0,0,'b','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1);
    legend([fh_finger,fh_normal,fh1,fh2,fh3], {"finger path", "surface normal",...
        "cf grad", "gqcnn grad", "self collision grad"},'location','southwest')
    end

axis equal
set(gca,'Clipping',"off")


end