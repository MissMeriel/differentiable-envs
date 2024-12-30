l = sym('l', [2 2], "real");
X = sym('X', [2 3], "real");
Y = sym('Y', [2 3], "real");
Z = sym('Z', [2 3], "real");
V = sym('V', [2,2,3], "real");
V = permute(V,[2,1,3]);
V = reshape(V,[4,3]);
W = sym('W', [3,3], "real");
C = sym('C', [1 3], "real");
Cs =  sym('Cs', [1 3], "real");
r = sym('r', [1], "real");
N = sym('N', [3,1], "real");
VN = sym('VN', [4,1], "real");
CN = sym('CN', [1,1], "real");
%% currently just used to verify


bary = [X(1,1) - X(1,3), X(1,2) - X(1,3), 0,0, X(1,3);
        Y(1,1) - Y(1,3), Y(1,2) - Y(1,3), 0,0, Y(1,3);
        Z(1,1) - Z(1,3), Z(1,2) - Z(1,3), 0,0, Z(1,3);
        0,0,X(2,1) - X(2,3), X(2,2) - X(2,3), X(2,3);
        0,0,Y(2,1) - Y(2,3), Y(2,2) - Y(2,3), Y(2,3);
        0,0,Z(2,1) - Z(2,3), Z(2,2) - Z(2,3), Z(2,3)]';

point_diff = [eye(3),-eye(3)]';

A_half = bary * point_diff;

A_with_const = A_half * A_half' ;

A = A_with_const(1:4,1:4);

%%

vector_with_const = [l(1,1), l(1,2), l(2,1), l(2,2), 1];

dist = vector_with_const * A_with_const * vector_with_const';

%%
A_jac = 0.5 * jacobian(jacobian(dist,vector_with_const(1:4)),vector_with_const(1:4));
isequal(A_jac,A);

%% triangle to triangle
point_1 = baryforward(l(1,:),[X(1,:);Y(1,:);Z(1,:)]);
point_2 = baryforward(l(2,:),[X(2,:);Y(2,:);Z(2,:)]);

dist_bary = sum((point_2 - point_1).^2);

[coeffs_ret_1,coeffs_ret_2]=coeffs(dist_bary,vector_with_const(1:4)');
A_bary = coeffs_ret_1([1, 2, 3, 4; 2, 6, 7, 8; 3, 7, 10, 11; 4, 8, 11, 13]) .* (0.5*(1-eye(4)) + eye(4))
linear_terms = coeffs_ret_1([5,9,12,14])'
const_term = coeffs_ret_1(15)

%% triangle to edge

point_2_edge = baryforward([l(2,1),-1/2],[X(2,:);Y(2,:);Z(2,:)],1/2);

dist_bary_edge = sum((point_2_edge - point_1).^2);

[coeffs_ret_1_edge,coeffs_ret_2_edge]=coeffs(dist_bary_edge,vector_with_const(1:4)');
A_bary_edge = coeffs_ret_1_edge([1,2,3;2,5,6;3,6,8]) .* (0.5*(1-eye(3)) + eye(3))
linear_terms_edge = coeffs_ret_1_edge([4,7,9])'
const_term_edge = coeffs_ret_1_edge(10)

%% triangle to vert

point_2_vert = baryforward([-1/3,-1/3],[X(2,:);Y(2,:);Z(2,:)]);

dist_bary_vert = sum((point_2_vert - point_1).^2);

[coeffs_ret_1_vert,coeffs_ret_2_vert]=coeffs(dist_bary_vert,vector_with_const(1:4)');
A_bary_vert = coeffs_ret_1_vert([1,2;2,4]) .* (0.5*(1-eye(2)) + eye(2))
linear_terms_vert = coeffs_ret_1_vert([3,5])'
const_term_vert = coeffs_ret_1_vert(6)

%% Test with made up triangle
triXY = [-1,0,0;0,-1,0;1,1,0]';
triXZ = [-1,0,0;0,0,-1;1,0,1]';
offset = [0,0,50]';
A_bary_fixed = subs(A_bary, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[triXY,triXZ + offset]);
linear_fixed = subs(linear_terms, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[triXY,triXZ + offset]);
ineqA = [-eye(4);[1,1,0,0;0,0,1,1]];
ineqB = [0,0,0,0,1,1];

[lambdas, dist_quadratic, exitflag, output, lambda_return] = quadprog(2*double(A_bary_fixed),double(linear_fixed),ineqA,ineqB);

dist_with_const_bary_point = subs(dist_bary,[[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[triXY,triXZ + offset]);
dist_with_const_bary = sqrt(double(subs(dist_with_const_bary_point,vector_with_const(1:4),lambdas')));

dist_quadratic_with_const = sqrt(dist_quadratic + double(subs(const_term, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[triXY,triXZ + offset])));

baryforward(lambdas(1:2),(triXY));
baryforward(lambdas(3:4),(triXZ + offset));

%% simplify into edge terms
POS = [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]];
edge_vectors = [POS(:,1)-POS(:,3),POS(:,2)-POS(:,3),POS(:,4)-POS(:,6),POS(:,5)-POS(:,6)];
triangle_corner_vectors = [POS(:,4:6)-POS(:,1:3)];
vectors_diff = [edge_vectors,triangle_corner_vectors];

arrayOfTerms = convertToEdgeVector(A_bary, linear_terms, const_term,POS, edge_vectors, vectors_diff, V, W, Cs, C);
A_diff = arrayOfTerms{1}
linear_diff = arrayOfTerms{2}
const_diff = arrayOfTerms{3}
point_sum_C = arrayOfTerms{4};
C_tri = -point_sum_C/3

arrayOfTerms = convertToEdgeVector(A_bary_edge, linear_terms_edge, const_term_edge,POS, edge_vectors, vectors_diff, V, W, Cs, C);
A_diff_edge = arrayOfTerms{1}
linear_diff_edge = arrayOfTerms{2}
const_diff_edge = arrayOfTerms{3}
point_sum_C_edge = arrayOfTerms{4};
C_edge = -point_sum_C_edge/3

arrayOfTerms = convertToEdgeVector(A_bary_vert, linear_terms_vert, const_term_vert,POS, edge_vectors, vectors_diff, V, W, Cs, C);
A_diff_vert = arrayOfTerms{1}
linear_diff_vert = arrayOfTerms{2}
const_diff_vert = arrayOfTerms{3}
point_sum_C_vert = arrayOfTerms{4};
C_vert = -point_sum_C_vert/3
%%
%% Get constraints
normal_1 = cross(edge_vectors(:,1),edge_vectors(:,2));

arrayOfTerms = generateConstraint(point_1, point_2, point_sum_C, normal_1,vectors_diff,vector_with_const,r,V, W, Cs, C,N,VN,CN);
A = arrayOfTerms{1}
b = arrayOfTerms{2}
Normal_diff_N = arrayOfTerms{3}
C_tri

arrayOfTerms = generateConstraint(point_1, point_2_edge, point_sum_C_edge, normal_1,vectors_diff,vector_with_const,r,V, W, Cs, C,N,VN,CN);
A_edge = arrayOfTerms{1}
b_edge = arrayOfTerms{2}
Normal_diff_N_edge = arrayOfTerms{3}
C_edge

arrayOfTerms = generateConstraint(point_1, point_2_vert, point_sum_C_vert, normal_1,vectors_diff,vector_with_const,r,V, W, Cs, C,N,VN,CN);
A_vert = arrayOfTerms{1}
b_vert = arrayOfTerms{2}
Normal_diff_N_vert = arrayOfTerms{3}
C_vert
%% Verify with triangles from visGraspScript, expects variables to exist
% equality
QPExpr = reshape(l',1,4)*A_bary*reshape(l',1,4)' + reshape(l',1,4) * linear_terms  + const_term;
isAlways(QPExpr == dist_bary)
%[coeffs_ret_1_test,coeffs_ret_2_test] = coeffs(QPExpr,vector_with_const(1:4)');

tr0 =  bar.v(bar.f.v((diststableFiltered.ind0(1)+1),:)',:)
tr1 =  bar.v(bar.f.v((diststableFiltered.ind1(1)+1),:)',:)
% test triangle
% tr0 = [    0.0081   -0.0251    0.0332
%     0.0453   -0.0381    0.0514
%     0.0132   -0.0304    0.0605];
tr0 = [tr0(2:3,:);tr0(1,:)]; % pytorch to symbolic math
% tr1 = [    0.0271    0.0065    0.0034
%     0.0108    0.0105    0.0072
%     0.0248    0.0080   -0.0025];
tr1 = [tr1(2:3,:);tr1(1,:)]; % pytorch to symbolic math
%l_double = [0, 1; 0.998, 0];
l_double = [diststableFiltered.b0_1(1),diststableFiltered.b0_2(1);diststableFiltered.b1_1(1),diststableFiltered.b1_2(1)];

cnst_double = double(subs(const_term, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[tr0',tr1']))
line_double = double(subs(linear_terms, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[tr0',tr1']))
quad_double = double(subs(A_bary, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[tr0',tr1']))
dist_double = double(subs(subs(dist_bary, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[tr0',tr1']),l,l_double))

tr0_point = baryforward(l_double(1,:), tr0')
tr1_point = baryforward(l_double(2,:), tr1')

ineqG = [-eye(4);[1,1,0,0;0,0,1,1]];
ineqh = [1/3,1/3,1/3,1/3,1/3,1/3];

quad_double = quad_double + eye(size(quad_double,1))* diststableFiltered.regularizer(1)

[lambdas, dist_quadratic, exitflag, output, lambda_return] = ...
    quadprog(2*quad_double,line_double,ineqG,ineqh)

tr0_point_solve = baryforward(lambdas(1:2), tr0')
tr1_point_solve = baryforward(lambdas(3:4), tr1')

dist_solve = sum((tr0_point_solve - tr1_point_solve).^2)
%% adding normal constraint
quad_double_normal = quad_double;
% quad_double_normal(5,5) = 0;%diststableFiltered.regularizer(1)

line_double_normal = line_double;
% line_double_normal(5) = 0

ineqG_normal = ineqG;
normal_double = double(subs(normal_1,[[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[tr0',tr1']));
normal_double = normal_double/norm(normal_double)
% ineqG_normal(:,5) = 0
A_points = subs(A, [V',W',Cs'/3],  [vectors_diff,-pointsum_diff/3]);
A_points = subs(A_points, VN,  vectors_diff(:,1:4)' * normal_double);
A_points = subs(A_points, N, normal_double)
eqA = double(subs(A_points, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[tr0',tr1']))


b_points = subs(b, [V',W',C'],  [vectors_diff,pointsum_diff/3]);
b_points = subs(b_points, CN,  pointsum_diff'/3 * normal_double);
b_points = subs(b_points, N, normal_double)
eqb = double(subs(b_points, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[tr0',tr1']))

%eqb = double(subs(pointsum_diff/3, [[X(1,:);Y(1,:);Z(1,:)],[X(2,:);Y(2,:);Z(2,:)]],[tr0',tr1']))

[lambdas_normal, dist_quadratic_normal, exitflag_normal, output_normal, lambda_return_normal] = ...
    quadprog(2*quad_double_normal,line_double_normal,ineqG_normal,ineqh,...
    eqA,eqb)

tr0_point_solve_normal = baryforward(lambdas_normal(1:2), tr0')
tr1_point_solve_normal = baryforward(lambdas_normal(3:4), tr1')

eq_error = eqA * lambdas_normal - eqb

dist_solve_normal = sum((tr0_point_solve_normal - tr1_point_solve_normal).^2)
%% plots on top of visGraspScript
plot3([tr0_point_solve_normal(1),tr1_point_solve_normal(1)],[tr0_point_solve_normal(2),tr1_point_solve_normal(2)],[tr0_point_solve_normal(3),tr1_point_solve_normal(3)],'.-','LineWidth',2)
plot3([tr0_point_solve(1),tr1_point_solve(1)],[tr0_point_solve(2),tr1_point_solve(2)],[tr0_point_solve(3),tr1_point_solve(3)],'.-','LineWidth',2)
quiver3(tr0_point_solve_normal(1),tr0_point_solve_normal(2),tr0_point_solve_normal(3),...
    normal_double(1)/20,normal_double(2)/20,normal_double(3)/20,'m','LineWidth',2)

%% 
% traditional
% function val = baryforward(lambda, points)
%     val = lambda(1)*(points(:,1)) + lambda(2)*(points(:,2)) + (1 - lambda(1) - lambda(2)) * points(:,3);
% end

% centered
function val = baryforward(lambda, points, offset)
    if nargin < 3
        offset = 1/3;
    end
    lambda = lambda + offset;
    val = lambda(1)*(points(:,1)) + lambda(2)*(points(:,2)) + (1 - lambda(1) - lambda(2)) * points(:,3);
end

function arrayOfTerms = convertToEdgeVector(A_bary, linear_terms, const_term, POS, edge_vectors, vectors_diff, V, W, Cs, C)


    unique_expr = [A_bary(triu(true(numel(linear_terms)))); linear_terms;const_term];
    
    if size(A_bary,1) == 4
        pointsum = [sum(POS(:,1:3),2),sum(POS(:,4:6),2)];
    elseif size(A_bary,1) == 3
        pointsum = [sum(POS(:,1:3),2),sum(POS(:,[4,6]),2) * 3/2];
    elseif size(A_bary,1) == 2
        pointsum = [sum(POS(:,1:3),2),3*POS(:,6)];
    end
    pointsum_diff = pointsum(:,1) - pointsum(:,2) ;
    diffexpr = subs(unique_expr, vectors_diff, [V',W']);
    A_diff = subs(A_bary, [vectors_diff], [V',W']);
    linear_diff = subs(linear_terms, [vectors_diff, pointsum_diff/3], [V',W',Cs'/3]);
    const_diff = subs(const_term, [vectors_diff,pointsum_diff/3], [V',W',Cs'/3]);
    
    % subs isn't smart about coefficients, so we subs again
    linear_diff =subs(linear_diff,Cs,-C*3);
    const_diff = subs(const_diff,Cs,-C*3);
    arrayOfTerms = {A_diff, linear_diff, const_diff, pointsum_diff, edge_vectors};
end

function arrayOfTerms = generateConstraint(point_1, point_2, point_sum_C, normal_1,vectors_diff,vector_with_const,r,V, W, Cs, C,N,VN,CN)
    
    
    %constraint = normal_1 * r + point_2 - point_1
    constraint = (point_2 - point_1)' * normal_1 * normal_1 - (point_2 - point_1);
    [coeffs_ret_1_constraint_1,coeffs_ret_2_constraint]=coeffs(constraint(1),[vector_with_const(1:4),r]');
    [coeffs_ret_1_constraint_2,coeffs_ret_2_constraint]=coeffs(constraint(2),[vector_with_const(1:4),r]');
    [coeffs_ret_1_constraint_3,coeffs_ret_2_constraint]=coeffs(constraint(3),[vector_with_const(1:4),r]');
    
    constraintMat = [coeffs_ret_1_constraint_1; coeffs_ret_1_constraint_2; coeffs_ret_1_constraint_3];
    
    Normal_diff = subs(normal_1, [vectors_diff,point_sum_C/3], [V',W',Cs'/3]);
    constraint_diff = subs(constraintMat, [vectors_diff,point_sum_C/3], [V',W',Cs'/3]);
    constraint_diff = subs(constraint_diff, Normal_diff, N);
    constraint_diff = subs(constraint_diff, V*N, VN);
    constraint_diff = subs(constraint_diff,Cs,C*3);
    constraint_diff = subs(constraint_diff, C*N, CN);
    A = constraint_diff(:,1:(end-1));
    b = -constraint_diff(:,end);
    arrayOfTerms = {A, b, Normal_diff};
end