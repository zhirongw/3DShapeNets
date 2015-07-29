function [new_TSDF] = sample2TSDF_fast(completed_samples, center, K, R, trans, halfWidth, R_cam, trans_cam, depth)
% Project completed samples to the surface specified by the new camera.
% completed_samples: input completions
% center: object center in world coordinate
% K: camera intrinsic
% R, trans: camera extrinsic in the original view
% R_cam, trans_cam: camera extrinsic in the next view
% halfWidth: object size along the x,y,z direction
% depth: depth map for the original view
% new_TSDF: new TSDF combining the original and the next view. (1 for
% surface, 0 for empty space, -1 for unknown voxels)

new_TSDF = zeros(size(completed_samples));

% camera parameters
ratio = 1;
imw = 640 * ratio; 
imh = 480 * ratio;
if isempty(K)
    fx_rgb = 5.19e+02 * ratio;
    fy_rgb = 5.19e+02 * ratio;
    cx_rgb = imw/2;
    cy_rgb = imh/2;
    K=[fx_rgb 0 cx_rgb; 0 fy_rgb cy_rgb; 0 0 1];
end

C = [0;0;0]; 
z_near = 0.3;
z_far_ratio = 1.2;
P = K * R * [eye(3), -C];
obj_center_cam = R * center + trans;
obj_center_fake = inv(R) * obj_center_cam; % This is not the true object center!

pad_len = 3;
data_size = size(completed_samples,2);
volume_size = data_size - 2 * pad_len;
mult = 5;

completed_samples = permute(completed_samples, [1,3,4,2]);
n = size(completed_samples,1);
 
xc = center(1); yc = center(2); zc = center(3);
cube_biggest_len = 2 * max(halfWidth);
s = cube_biggest_len / (volume_size - 1);

xmin = xc - cube_biggest_len / 2 - s * pad_len; xmax = xc + cube_biggest_len / 2 + s * pad_len;
ymin = yc - cube_biggest_len / 2 - s * pad_len; ymax = yc + cube_biggest_len / 2 + s * pad_len;
zmin = zc - cube_biggest_len / 2 - s * pad_len; zmax = zc + cube_biggest_len / 2 + s * pad_len;

gridDim = [xmin,ymin,zmin,xmax,ymax,zmax];
stepSize = s;
if numel(stepSize) == 1, stepSize = repmat(stepSize,[1 3]); end
            
depth_size = [480,640];

gridSize_x = round((gridDim(4)-gridDim(1))/stepSize(1)+1);
gridSize_y = round((gridDim(5)-gridDim(2))/stepSize(2)+1);
gridSize_z = round((gridDim(6)-gridDim(3))/stepSize(3)+1);
[X,Y,Z] = ndgrid(1:gridSize_x,1:gridSize_y,1:gridSize_z);
gridCoord = bsxfun(@times,[X(:) Y(:) Z(:)]-1,stepSize(1:3));
gridCoord = bsxfun(@plus,gridCoord,gridDim(1:3));

for i = 1 : n
    isOccupied = (squeeze(completed_samples(i,:,:,:))==1);
    this_gridCoord = gridCoord(isOccupied,:);
    [vertices, faces] = create_mesh(this_gridCoord, stepSize);
    vertices = scalePoints(vertices, obj_center_fake, [1,1,1]');
    
    result = RenderMex(P, imw, imh, [vertices(1,:);vertices(2,:);vertices(3,:)], uint32(faces))';
    depth_new = z_near./(1-double(result)/2^32);
    maxDepth = 10;
    cropmask = (depth_new < z_near) | (depth_new > z_far_ratio * maxDepth);
    depth_new(cropmask) = NaN;
    
    R_cam{end+1} = R; trans_cam{end+1} = trans; depth{end+1} = depth_new;
    [this_gridDists, ~] = TSDF(depth, K, center, R_cam, trans_cam, volume_size * mult, pad_len * mult, halfWidth, [1,1]);
    R_cam(end) = []; trans_cam(end) = []; depth(end) = [];
    
    old_TSDF = cubicle2col(this_gridDists, mult);
    surface_num = sum((old_TSDF < 1 & old_TSDF > -1),1);
    out_num = sum((old_TSDF == 1),1);
    in_num = sum((old_TSDF == -1),1);

    surface_index = (surface_num > 0 & in_num > 0 & out_num > 0) | surface_num > 1;
    out_index = (out_num >= in_num) & ~surface_index;

    old_TSDF = - ones([data_size,data_size,data_size], 'single');
    old_TSDF(surface_index) = 1;
    old_TSDF(out_index) = 0;
    
    new_TSDF(i,:,:,:) = old_TSDF;
end

function coornew = scalePoints(coor, center, size)
minv = min(coor, [], 2);
maxv = max(coor, [], 2);
oldCenter = (minv+maxv)/2;
oldSize = maxv - minv;
scale = min(size./ oldSize);
coornew = bsxfun(@plus, scale * coor, center-scale*oldCenter);

function [vertices, faces] = create_mesh(points, stepSize)
n = size(points,1);
vertices = zeros(8*n,3);
faces = zeros(12*n,3);

const_offset = [1,1,1;1,1,-1,;1,-1,1;1,-1,-1;-1,1,1;-1,1,-1;-1,-1,1;-1,-1,-1];
step_offset = bsxfun(@times, const_offset, 0.5 * stepSize);

for v = 1 : 8
    vertices((v-1)*n+1:v*n,:) = bsxfun(@plus, points, step_offset(v,:));
end

faces(1:n,:) = [1:n; n+1:2*n; 2*n+1:3*n]';
faces(n+1:2*n,:) = [n+1:2*n; 2*n+1:3*n; 3*n+1:4*n]';
faces(2*n+1:3*n,:) = [1:n; n+1:2*n; 4*n+1:5*n]';
faces(3*n+1:4*n,:) = [n+1:2*n; 4*n+1:5*n; 5*n+1:6*n]';
faces(4*n+1:5*n,:) = [n+1:2*n; 3*n+1:4*n; 5*n+1:6*n]';
faces(5*n+1:6*n,:) = [3*n+1:4*n; 5*n+1:6*n; 7*n+1:8*n]';
faces(6*n+1:7*n,:) = [2*n+1:3*n; 3*n+1:4*n; 6*n+1:7*n]';
faces(7*n+1:8*n,:) = [3*n+1:4*n; 6*n+1:7*n; 7*n+1:8*n]';
faces(8*n+1:9*n,:) = [4*n+1:5*n; 5*n+1:6*n; 6*n+1:7*n]';
faces(9*n+1:10*n,:) = [5*n+1:6*n; 6*n+1:7*n; 7*n+1:8*n]';
faces(10*n+1:11*n,:) = [1:n; 2*n+1:3*n; 4*n+1:5*n]';
faces(11*n+1:12*n,:) = [2*n+1:3*n; 4*n+1:5*n; 6*n+1:7*n]';
faces(:,4) = faces(:,1);

faces = faces - 1;
vertices = vertices';
faces = faces';

