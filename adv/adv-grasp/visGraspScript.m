bar = readObj('data/bar_clamp.obj')
%%
figure;
hold on;
trisurf(bar.f.v,bar.v(:,1),bar.v(:,2),bar.v(:,3))
alpha 0.5
for i = 1:size(bar.f.v,1)
    triangle = bar.v(bar.f.v(i,:)',:);
    triangleCenter = mean(triangle,1);
    dir1 = triangle(2,1:3)-triangle(1,1:3);
    dir1 = dir1/norm(dir1);
    dir2 = triangle(3,1:3)-triangle(1,1:3);
    dir2 = dir2/norm(dir2);
    text(triangleCenter(1),triangleCenter(2),triangleCenter(3),num2str(i));
    normal = cross(dir1,dir2)*0.05/2;
    quiver3(triangleCenter(1),triangleCenter(2),triangleCenter(3),normal(1),normal(2),normal(3),'b','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)
end
%pcshow(bar.v(:,1:3))
ray_o = [[-0.0472,  0.0167, -0.0634];
        [ 0.0028,  0.0167, -0.0634]];
ray_d = [[ 1.,  0.,  0.]
        [-1., -0., -0.]] * 0.05/2;
quiver3(ray_o(:,1),ray_o(:,2),ray_o(:,3),...
    ray_d(:,1),ray_d(:,2),ray_d(:,3),'r','LineWidth',2,'AutoScale','off' ,'MaxHeadSize',1)