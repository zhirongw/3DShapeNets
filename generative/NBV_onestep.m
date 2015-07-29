function [prediction_o, prediction_next, NBV_v, RC_v, RAND_v, FUR_v]= NBV_onestep(model, filename, angle_inc)
% Next-Best-View for one step. Given a depth map, calculate the TSDF to
% reconstruct the 3D partial surface and other unknown spaces. Then decide
% the NBV with the least uncertainty from a set of possible camera
% actions(Rt matrix).

% filename: 3D object mesh model to be processed.
% volume_size: the size of volumetric representation(42).
% pad_len: padding size(3).
% angle_inc: rotation increment of each 3D mesh model(360).
% prediction_o: original class prediction of the first view.
% prediction_next: class prediction of the next num_v possible views.
% NBV_v: entropy_based NBV from num_v views.
% RC_v: reconstruction based NBV from num_v views.
% RAND_v: random selection from num_v views.
% FUR_v: furtheset away selection from num_v views.

rng('shuffle');
addpath voxelization;
addpath 3D;

mult = 5;
num_v = 8;
prediction_next = zeros(num_v, model.classes);

data_size = model.volume_size + 2 * model.pad_size;

depth = cell(1);
R_cam = cell(1); trans_cam = cell(1);
for viewpoint = 1 : 360/angle_inc
    
    obj_center = [0,0,-1.5]'; % object coordinate in camera system
    camera_center = [0,0,0]'; camera_direction = [0,0,1]';
    
    axis3d = cross(camera_direction, obj_center - camera_center);
    if all(axis3d == 0)
       axis3d = cross(camera_direction, [1,0,0]'); 
    end
    angle = atan2(norm(cross(camera_direction, obj_center - camera_center)),dot(camera_direction, obj_center - camera_center));
    axis_angle = axis3d / norm(axis3d) * (-angle);
    R_o = AngleAxis2RotationMatrix(axis_angle); trans_o = [0,0,0]';
    obj_center_cam = R_o * obj_center;
    
    [depth_new, K, crop] = off2im(filename, 1, (viewpoint - 1) * angle_inc * pi / 180, R_o, obj_center(1), obj_center(2), obj_center(3), [1;1;1], 0, 0);
    depth{1} = depth_new; R_cam{1} = R_o; trans_cam{1} = trans_o;
    [gridDists, halfWidth] = TSDF(depth, K, obj_center, R_cam, trans_cam, volume_size * mult, pad_len * mult, [], crop);

    old_TSDF = cubicle2col(gridDists, mult);
    surface_num = sum((old_TSDF < 1 & old_TSDF > -1),1);
    out_num = sum((old_TSDF == 1),1);
    in_num = sum((old_TSDF == -1),1);

    surface_index = (surface_num > 0 & in_num > 0 & out_num > 0) | surface_num > 1;
    out_index = (out_num >= in_num) & ~surface_index;

    old_TSDF = - ones([data_size,data_size,data_size], 'single');
    old_TSDF(surface_index) = 1;
    old_TSDF(out_index) = 0;
        
    param = []; param.nparticles = 50; param.earlyStop = false;  param.epochs = 50;
    [original_entropy, completed_samples, prediction_o]= get_recognition_entropy(model, permute(old_TSDF,[3,1,2]), param);
    fprintf('The original entropy: %f\n', original_entropy);
    %------------
    % k-means cluster the samples
    completed_samples = reshape(completed_samples, param.nparticles, []);
    k_clusters = 5;
    [idx, completed_samples] = kmeans(completed_samples, k_clusters, 'emptyaction', 'drop');
    [~,~,idx] = unique(idx); % map the idx to the lower 1 - k numbers
    k_clusters = length(unique(idx));
    completed_samples(isnan(completed_samples)) = [];
    completed_samples = reshape(completed_samples, [k_clusters, model.layers{1}.layerSize]);
    sample_weight = sum(full(sparse(idx, 1:param.nparticles, 1)), 2) / param.nparticles;
    %------------
    completed_samples = single(completed_samples > 0.1);

    NBV_entropy = zeros(num_v,1);
    RC_freespace = zeros(num_v,1);
    FUR_dis = zeros(num_v,1);
    [R, trans] = get_Rt(obj_center', R_o, trans_o, norm(obj_center), num_v);
    for v = 1 : num_v
        R_world = R{v} * R_o; trans_world = R{v} * trans_o + trans{v};
        
        % NBV predictions
        view_TSDF = sample2TSDF_fast(completed_samples, obj_center, K, R_world, trans_world, halfWidth, R_cam, trans_cam, depth);

        entropy = zeros(size(view_TSDF,1),1);
        param = []; param.nparticles = 50; param.earlyStop = true; param.epochs = 20;
        for h = 1 : size(view_TSDF,1)
            entropy(h) = get_recognition_entropy(model, permute(squeeze(view_TSDF(h,:,:,:)),[3,1,2]), param);
        end
        NBV_entropy(v) = sample_weight' * entropy;
        fprintf('\t the %d view entropy: %f\n', v, NBV_entropy(v));
        
        % RC prediction
        view_gridDists = get_next_view(old_TSDF, obj_center, K, R_world, trans_world, halfWidth, volume_size, pad_len, crop);
        RC_freespace(v) = sum(view_gridDists(:)==-2);
        
        % FUR prediction
        FUR_dis(v) = norm(- inv(R_world) * trans_world);
        
        % physical move : ground truth
        obj_center_cam = R_world * obj_center + trans_world;
        obj_center_fake = inv(R_world) * obj_center_cam; % This is not the true object center!
        [depth_new, K, crop] = off2im(filename, 1, (viewpoint - 1) * angle_inc * pi / 180, R_world, obj_center_fake(1), obj_center_fake(2), obj_center_fake(3), [1;1;1], 0, 0);
        depth{end+1} = depth_new; R_cam{end+1} = R_world; trans_cam{end+1} = trans_world;
        [gridDists, halfWidth] = TSDF(depth, K, obj_center, R_cam, trans_cam, volume_size * mult, pad_len * mult, [], crop); % points3d is in world coordinate

        old_TSDF = cubicle2col(gridDists, mult);
        surface_num = sum((old_TSDF < 1 & old_TSDF > -1),1);
        out_num = sum((old_TSDF == 1),1);
        in_num = sum((old_TSDF == -1),1);

        surface_index = (surface_num > 0 & in_num > 0 & out_num > 0) | surface_num > 1;
        out_index = (out_num >= in_num) & ~surface_index;

        old_TSDF = - ones([data_size,data_size,data_size], 'single');
        old_TSDF(surface_index) = 1;
        old_TSDF(out_index) = 0;
        
        param = []; param.nparticles = 50; param.earlyStop = false; param.epochs = 50;
        [~, ~, this_label]= get_recognition_entropy(model, permute(old_TSDF,[3,1,2]), param);
        prediction_next(v, :) = this_label;
        
        depth(end) = []; R_cam(end) = []; trans_cam(end) = [];
    end
    % decide the NBV
    [~, NBV_v] = min(NBV_entropy);
    % decide the RC
    [~, RC_v] = max(RC_freespace);
    %decide the FUR
    [~, FUR_v] = max(FUR_dis);
    % decide random
    RAND_v = randi(num_v);
end

function [R_dense, translation_dense] = get_Rt(obj_center_world, R_o, trans_o, r, N)
% The returned R, translation are the transformations between adjacent
% views.
camera_direction = [0,0,1]; % always let the camera target at the object
rng('shuffle');
p = rand(N,3) - 0.5;
p = r .* bsxfun(@rdivide, p, sqrt(sum(p.^2,2)));
p = bsxfun(@plus, p, obj_center_world);

R_dense = cell(1); translation_dense = cell(1);

for v = 1 : N
    rng('shuffle');
    jitter = (rand(1,3) - 0.5) * 2.5;
    temp = R_o * (obj_center_world + jitter - p(v,:))' + trans_o;
    translation = R_o * p(v,:)' + trans_o;
    
    axis = cross(camera_direction, temp);
    if all(axis == 0)
       axis = cross(camera_direction, [1,0,0]');
    end
    angle = atan2(norm(cross(camera_direction, temp)),dot(camera_direction, temp));
    axis_angle = axis / norm(axis) * (-angle);
    R = AngleAxis2RotationMatrix(axis_angle);
    translation = - R * translation;
    
    R_dense{v} = R;
    translation_dense{v} = translation;
end
