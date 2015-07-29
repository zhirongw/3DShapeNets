function [all_gridDists, halfWidth] = TSDF(depth, K, center, R, trans, volume_size, pad_len, halfWidth, crop)
% This TSDF merges multiple depth map (and Rt)
% altogether into one TSDF. TSDF: 1 for object surface, 0 for empty spaces,
% -1 for unknown voxels.

% depth: depth map for the original view
% K: camera intrinsic
% center: object center in world coordinate
% R, trans: camera extrinsic in the original view
% volume_size: the size of volumetric representation(42).
% pad_len: padding size(3).
% halfWidth: object size along the x,y,z direction
% all_gridDists: target TSDF.

assert(length(depth) == length(R) && length(R) == length(trans));
nViews = length(depth);
xc = center(1); yc = center(2); zc = center(3);
if isempty(halfWidth)
    halfWidth = zeros(1,3);
    for i = 1 : nViews
        [~,points3d] = read_3d_pts_general(depth{i},K,size(depth{i}),[],crop);

        points3d = (inv(R{i}) * bsxfun(@minus, points3d', trans{i}))';

        xhalfwidth = max(abs(points3d(:,1) - xc));
        yhalfwidth = max(abs(points3d(:,2) - yc));
        zhalfwidth = max(abs(points3d(:,3) - zc));
        halfWidth_temp = [xhalfwidth, yhalfwidth, zhalfwidth];
        halfWidth = max(halfWidth, halfWidth_temp);
    end
end
cube_biggest_len = 2 * max(halfWidth);
s = cube_biggest_len / (volume_size - 1);

xmin = xc - cube_biggest_len / 2 - s * pad_len; xmax = xc + cube_biggest_len / 2 + s * pad_len;
ymin = yc - cube_biggest_len / 2 - s * pad_len; ymax = yc + cube_biggest_len / 2 + s * pad_len;
zmin = zc - cube_biggest_len / 2 - s * pad_len; zmax = zc + cube_biggest_len / 2 + s * pad_len;

gridDim = [xmin,ymin,zmin,xmax,ymax,zmax];
stepSize = s; mu = s;
if numel(stepSize) == 1, stepSize = repmat(stepSize,[1 3]); end

all_gridDists = -1 * ones(volume_size+2*pad_len,volume_size+2*pad_len,volume_size+2*pad_len);
for i = 1 : nViews
    gridDists = - mu * ones(volume_size+2*pad_len,volume_size+2*pad_len,volume_size+2*pad_len);
    [~,points3d] = read_3d_pts_general(depth{i},K,size(depth{i}),[],crop);
    points3d = (inv(R{i}) * bsxfun(@minus, points3d', trans{i}))';

    [X,Y,Z] = ndgrid(1:round((gridDim(4)-gridDim(1))/stepSize(1)+1),1:round((gridDim(5)-gridDim(2))/stepSize(2)+1),1:round((gridDim(6)-gridDim(3))/stepSize(3))+1);
    gridIdx = sub2ind(size(X),X,Y,Z); 
    gridIdx = gridIdx(:);

    gridCoord = bsxfun(@times,[X(:) Y(:) Z(:)]-1,stepSize(1:3));
    gridCoord = bsxfun(@plus,gridCoord,gridDim(1:3));
    clear X Y Z;

    % projection
    % model and Kinect data diff
    [gridProj,gridProjDepth] = project3dPtsTo2d(gridCoord, K, R{i}, trans{i}, crop);   
    gridProj = round(gridProj);

    % remove invalid projections
    isValid = gridProj(:,1) >= 1 & gridProj(:,1) <= size(depth{i},2) & ...
              gridProj(:,2) >= 1 & gridProj(:,2) <= size(depth{i},1) & ...
              gridProjDepth > 0;
    gridIdx = gridIdx(isValid);
    gridCoord = gridCoord(isValid,:);
    gridProj = gridProj(isValid,:);
    gridProjDepth = gridProjDepth(isValid);
    gridDists(gridIdx) = mu;

    % remove point that has no depth
    gridProjIdx = sub2ind(size(depth{i}),gridProj(:,2),gridProj(:,1));
    isValid = ~isnan(depth{i}(gridProjIdx)) & depth{i}(gridProjIdx)~=0;
    gridIdx = gridIdx(isValid);
    gridCoord = gridCoord(isValid,:);
    gridProjIdx = gridProjIdx(isValid,:);
    gridProjDepth = gridProjDepth(isValid);

    % compute distance
    dists = sqrt(sum((gridCoord-points3d(gridProjIdx,:)).^2,2));
    dists(gridProjDepth > depth{i}(gridProjIdx)) = -dists(gridProjDepth > depth{i}(gridProjIdx));
    gridDists(gridIdx) = dists;
    gridDists = gridDists / mu;

    gridDists = max(-1,gridDists);
    gridDists = min(1,gridDists);

    all_gridDists = merge_TSDF(all_gridDists, gridDists); 
end
    
function all_gridDists = merge_TSDF(all_gridDists, gridDists)
%all_gridDists = max(all_gridDists, gridDists);

temp_gridDists = all_gridDists;
% leave the +1 ones unchanged

% deal with the boundaries
knownVoxels = (gridDists > - 1 & gridDists < 1) & (temp_gridDists > - 1 & temp_gridDists < 1);
changeVoxels = ((temp_gridDists < 0 & gridDists > 0) | (temp_gridDists > gridDists & gridDists > 0) | (temp_gridDists < gridDists & gridDists < 0)) & knownVoxels  ;
all_gridDists(changeVoxels) = gridDists(changeVoxels);

% update the visible ones
all_gridDists(temp_gridDists==-1) = gridDists(temp_gridDists==-1);
    
    
