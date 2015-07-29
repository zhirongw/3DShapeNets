function new_gridDists = get_next_view(gridDists, center, K, R, trans, halfWidth, volume_size, pad_len, crop)
% Given an original gridDists(TSDF), and the new camera parameters
% (R,trans), the function computes the new voxels that will be observed in
% the new view. The new voxels will be marked with TSDF value -2.
% The function is only used for visualization and maximum visibility
% approach.

% gridDists: original TSDF, 1 for surfaces, 0 for empty spaces, -1 for
% unknown voxels.
% center: object position in the world coordinate.
% K: camera intrinsic matrix.
% R, trans: next view camera parameters
% halfWidth: the observed half length along x,y,z world coordinate.
% Returned from TSDF.m
% volume_size: the size of volumetric representation(42).
% pad_len: padding size(3).
% crop: always set it to 1 for now. Sorry.


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
new_gridDists = gridDists;

gridSize_x = round((gridDim(4)-gridDim(1))/stepSize(1)+1);
gridSize_y = round((gridDim(5)-gridDim(2))/stepSize(2)+1);
gridSize_z = round((gridDim(6)-gridDim(3))/stepSize(3)+1);
[X,Y,Z] = ndgrid(1:gridSize_x,1:gridSize_y,1:gridSize_z);
gridIdx = sub2ind(size(X),X,Y,Z); 
gridIdx = gridIdx(:);

gridCoord = bsxfun(@times,[X(:) Y(:) Z(:)]-1,stepSize(1:3));
gridCoord = bsxfun(@plus,gridCoord,gridDim(1:3));
%gridDists = mu * ones(size(X));

X = 1 : (gridSize_x + 1);
Y = 1 : (gridSize_y + 1);
Z = 1 : (gridSize_z + 1);
edgeCoord = bsxfun(@times, [X(:) Y(:) Z(:)]-1, stepSize(1:3));
edgeCoord = bsxfun(@plus, edgeCoord, gridDim(1:3)-0.5*stepSize(1:3));
clear X Y Z;

% projection
% model and Kinect data diff
[gridProj,gridProjDepth] = project3dPtsTo2d(gridCoord, K, R, trans, crop);   
gridProj = round(gridProj);

% remove invalid projections
isValid = gridProj(:,1) >= 1 & gridProj(:,1) <= depth_size(2) & ...
          gridProj(:,2) >= 1 & gridProj(:,2) <= depth_size(1) & ...
          gridProjDepth > 0;
gridIdx = gridIdx(isValid);
gridCoord = gridCoord(isValid,:);
gridProjDepth = gridProjDepth(isValid);
gridDists = gridDists(:);
gridDists = gridDists(isValid);
unobservedIdx = find(gridDists == -1);

trans = - inv(R) * trans; % This is the camera position in world coordinate.
while ~isempty(unobservedIdx)
    %fprintf('%d\n',length(unobservedIdx));
    the_unobserved = unobservedIdx(1);
    unobserved = gridCoord(the_unobserved,:);
    numerator = unobserved - (trans');
    
    final_index = gridIdx(the_unobserved);
    
    intersectionIdx = zeros(size(new_gridDists));
    intersectionIdx = intersectionIdx(:);
    intersectionIdx(final_index) = 1; % add the current point to the intersection list. Should need not to do this, just avoid numerical accident.
    % for x slices.
    if numerator(1) ~= 0
        intersection_y = (edgeCoord(:,1) - (trans(1))) / numerator(1) * numerator(2) + (trans(2));
        intersection_z = (edgeCoord(:,1) - (trans(1))) / numerator(1) * numerator(3) + (trans(3));
        intersection_y = ceil((intersection_y - gridDim(2) + 0.5 * stepSize(2)) / stepSize(2));
        intersection_z = ceil((intersection_z - gridDim(3) + 0.5 * stepSize(3)) / stepSize(3));
        intersection_x = (1 : gridSize_x)';

        temp_y = intersection_y(1:end-1); temp_z = intersection_z(1:end-1);
        interValid = (temp_y >= 1 & temp_y <= gridSize_y & temp_z >=1 & temp_z <= gridSize_z);
        intersectionIdx(sub2ind(size(new_gridDists), intersection_x(interValid), temp_y(interValid), temp_z(interValid))) = 1;

        temp_y = intersection_y(2:end); temp_z = intersection_z(2:end);
        interValid = (temp_y >= 1 & temp_y <= gridSize_y & temp_z >=1 & temp_z <= gridSize_z);
        intersectionIdx(sub2ind(size(new_gridDists), intersection_x(interValid), temp_y(interValid), temp_z(interValid))) = 1;
    end
    % for y slices.
    if numerator(2) ~= 0
        intersection_x = (edgeCoord(:,2) - (trans(2))) / numerator(2) * numerator(1) + (trans(1));
        intersection_z = (edgeCoord(:,2) - (trans(2))) / numerator(2) * numerator(3) + (trans(3));
        intersection_x = ceil((intersection_x - gridDim(1) + 0.5 * stepSize(1)) / stepSize(1));
        intersection_z = ceil((intersection_z - gridDim(3) + 0.5 * stepSize(3)) / stepSize(3));
        intersection_y = (1 : gridSize_y)';

        temp_x = intersection_x(1:end-1); temp_z = intersection_z(1:end-1);
        interValid = (temp_x >= 1 & temp_x <= gridSize_x & temp_z >=1 & temp_z <= gridSize_z);
        intersectionIdx(sub2ind(size(new_gridDists), temp_x(interValid), intersection_y(interValid),  temp_z(interValid))) = 1;

        temp_x = intersection_x(2:end); temp_z = intersection_z(2:end);
        interValid = (temp_x >= 1 & temp_x <= gridSize_x & temp_z >=1 & temp_z <= gridSize_z);
        intersectionIdx(sub2ind(size(new_gridDists), temp_x(interValid), intersection_y(interValid), temp_z(interValid))) = 1;
    end
    % for z slices.
    if numerator(3) ~= 0
        intersection_x = (edgeCoord(:,3) - (trans(3))) / numerator(3) * numerator(1) + (trans(1));
        intersection_y = (edgeCoord(:,3) - (trans(3))) / numerator(3) * numerator(2) + (trans(2));
        intersection_x = ceil((intersection_x - gridDim(1) + 0.5 * stepSize(1)) / stepSize(1));
        intersection_y = ceil((intersection_y - gridDim(2) + 0.5 * stepSize(2)) / stepSize(2));
        intersection_z = (1 : gridSize_z)';

        temp_x = intersection_x(1:end-1); temp_y = intersection_y(1:end-1);
        interValid = (temp_x >= 1 & temp_x <= gridSize_x & temp_y >=1 & temp_y <= gridSize_y);
        intersectionIdx(sub2ind(size(new_gridDists), temp_x(interValid), temp_y(interValid), intersection_z(interValid))) = 1;

        temp_x = intersection_x(2:end); temp_y = intersection_y(2:end);
        interValid = (temp_x >= 1 & temp_x <= gridSize_x & temp_y >=1 & temp_y <= gridSize_y);
        intersectionIdx(sub2ind(size(new_gridDists), temp_x(interValid), temp_y(interValid), intersection_z(interValid))) = 1;
    end
    intersectionIdx = intersectionIdx(isValid);
    in_ray_idx = find(intersectionIdx == 1);
    ray_depth = gridProjDepth(in_ray_idx,:);
    [~, sort_ind]= sort(ray_depth, 'ascend');

    the_unobserved_in_ray_idx = find(in_ray_idx == the_unobserved);
    the_unobserved_sort_in_ray_idx = find(sort_ind == the_unobserved_in_ray_idx);
 
    if all(new_gridDists(gridIdx(in_ray_idx(sort_ind(1:the_unobserved_sort_in_ray_idx-1)))) ~= 1)
        new_gridDists(final_index) = -2;
    end
    unobservedIdx(1) = [];
    
end
